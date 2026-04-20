"""
测试执行引擎模块

负责执行测试用例、收集结果、评估准确度，
并生成 Markdown 报告和 JSON 结果文件。

支持跨多个模型执行测试（如 seed-1-6-vision vs 1-5-vision-pro）。
"""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .client import ArkClient
from .config import (
    COORDINATE_SCALE,
    DEFAULT_MODELS,
    MODELS,
    OUTPUT_DIR,
)
from .ground_truth import GTItem, get_ground_truth
from .test_cases import ALL_TEST_CASES, TestCase, TestCategory


# 请求间隔，避免限流（仅串行模式下使用；并行模式下不再 sleep）
REQUEST_INTERVAL_SECONDS = 2

# 坐标评估：点落在 GT bbox 外的容差（0-1000 空间）
# 用于 point-in-bbox 判断时对 bbox 做微扩（允许轻微偏移仍视为命中）
POINT_TOLERANCE = 5

# IoU 阈值：预测 bbox 与 GT bbox 的 IoU 超过此值视为命中
IOU_HIT_THRESHOLD = 0.3

# 并行执行配置
DEFAULT_PARALLEL_WORKERS = 4
DEFAULT_TASK_TIMEOUT_SECONDS = 120


class TestRunner:
    """测试执行器（支持多模型）"""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.client = ArkClient()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict[str, Any]] = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 用于并行模式下保护 self.results 与日志顺序
        self._lock = threading.Lock()

    # ========================================================================
    # 核心执行方法
    # ========================================================================
    def run_single_test(
        self,
        test_case: TestCase,
        model_name: str,
        timeout: float | None = None,
        silent: bool = False,
    ) -> dict[str, Any]:
        """
        在指定模型上执行单个测试用例。

        Args:
            timeout: HTTP 层单次请求超时秒数；超时抛 openai.APITimeoutError
            silent: 并行模式下可关闭详细打印，仅保留精简日志
        """
        if not silent:
            self._print_test_header(test_case, model_name)

        # 构建消息：
        # - gui-plus 等模型需要注入 system prompt（按 ModelConfig 决定）
        # - prompt 中的 {COORD_FMT} 占位符按模型 output_format 填充官方格式
        # - multi_target 用例使用批量输出模板
        messages = self.client.build_messages(
            prompt=test_case.prompt,
            image_path=test_case.image_path,
            model_cfg=MODELS.get(model_name),
            multi_target=test_case.multi_target,
        )

        # 调用模型
        if not silent:
            print("正在调用模型...")
        try:
            response = self.client.call(
                messages=messages,
                model_name=model_name,
                enable_thinking=test_case.enable_thinking,
                timeout=timeout,
            )
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            is_timeout = "Timeout" in type(e).__name__
            status = "TIMEOUT" if is_timeout else "ERROR"
            print(f"[{status}] {model_name} × {test_case.id}"
                  f"{'(thinking)' if test_case.enable_thinking else ''} 失败: {err_msg}")
            error_result = self._build_error_result(
                test_case, model_name, err_msg, status=status
            )
            with self._lock:
                self.results.append(error_result)
            return error_result

        # 输出响应摘要
        if not silent:
            self._print_response_summary(response)

        # 评估结果
        evaluation = self._evaluate(test_case, response, silent=silent)

        # 定位测试：标记坐标到图片
        marked_image_path = None
        if test_case.category == TestCategory.POSITIONING:
            coordinates = self._extract_coordinates(response["content"])
            if coordinates:
                marked_image_path = self._mark_coordinates(
                    test_case.image_path, coordinates, test_case.id, model_name
                )

        result = {
            "model": model_name,
            "test_id": test_case.id,
            "test_name": test_case.name,
            "category": test_case.category.value,
            "enable_thinking": test_case.enable_thinking,
            "thinking_applied": response.get("thinking_applied", False),
            "status": "OK",
            "response_time": response["response_time"],
            "usage": response["usage"],
            "content": response["content"],
            "thinking_content": response["thinking_content"],
            "evaluation": evaluation,
            "marked_image": str(marked_image_path) if marked_image_path else None,
        }

        with self._lock:
            self.results.append(result)
        return result

    def run_all(
        self,
        test_cases: list[TestCase] | None = None,
        model_names: list[str] | None = None,
        thinking_matrix: bool = False,
        parallel: int = DEFAULT_PARALLEL_WORKERS,
        task_timeout: float = DEFAULT_TASK_TIMEOUT_SECONDS,
    ):
        """
        在所有指定模型上执行所有测试用例（默认并行）。

        Args:
            thinking_matrix: 支持 thinking 的模型额外跑一份 enable_thinking=True 的副本
            parallel: 并发任务数（>=2 启用并行；==1 退化为串行）
            task_timeout: 单任务 HTTP 超时秒数，超时后标记为 TIMEOUT 失败
        """
        from dataclasses import replace as _dc_replace

        if test_cases is None:
            test_cases = ALL_TEST_CASES
        if model_names is None:
            model_names = list(DEFAULT_MODELS)

        # 构建 (model, test_case) 执行列表
        exec_plan: list[tuple[str, TestCase]] = []
        for model_name in model_names:
            model_cfg = MODELS.get(model_name)
            for tc in test_cases:
                if tc.enable_thinking and (
                    not model_cfg or not model_cfg.supports_thinking
                ):
                    print(f"[跳过] {model_name} 不支持深度思考，跳过用例 {tc.id}")
                    continue
                exec_plan.append((model_name, tc))
                # Thinking matrix：支持 thinking 的模型为非 thinking 用例补一份 True 副本
                if (
                    thinking_matrix
                    and model_cfg
                    and model_cfg.supports_thinking
                    and not tc.enable_thinking
                ):
                    tc_think = _dc_replace(tc, enable_thinking=True)
                    exec_plan.append((model_name, tc_think))

        total = len(exec_plan)
        parallel = max(1, parallel)

        print(f"\n{'#' * 60}")
        print(f"  GUI 模型性能测试")
        print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  待测模型: {', '.join(model_names)}")
        print(f"  总执行数: {total} ({len(test_cases)} 用例 × {len(model_names)} 模型)")
        print(f"  并发数:   {parallel}   单任务超时: {task_timeout}s")
        print(f"{'#' * 60}")

        wall_start = time.time()

        if parallel == 1:
            # 串行路径（保留详细日志）
            for i, (model_name, tc) in enumerate(exec_plan):
                print(f"\n[{i + 1}/{total}]")
                self.run_single_test(tc, model_name, timeout=task_timeout)
                if i < total - 1:
                    time.sleep(REQUEST_INTERVAL_SECONDS)
        else:
            # 并行路径（silent=True，输出精简；失败/超时按 [i/total] 打印）
            done_counter = {"n": 0}

            def _task(idx: int, model_name: str, tc: TestCase) -> dict:
                r = self.run_single_test(
                    tc, model_name, timeout=task_timeout, silent=True
                )
                with self._lock:
                    done_counter["n"] += 1
                    n = done_counter["n"]
                    status = r.get("status", "?")
                    elapsed = r.get("response_time", "-")
                    coord = (r.get("evaluation") or {}).get("coordinate_eval") or {}
                    hit = coord.get("point_hit_rate")
                    cov = coord.get("gt_coverage_rate")
                    acc = (r.get("evaluation") or {}).get("accuracy")
                    metric_parts = []
                    if hit is not None:
                        metric_parts.append(f"Pt {hit*100:.0f}%")
                    if cov is not None:
                        metric_parts.append(f"Cov {cov*100:.0f}%")
                    if acc is not None and hit is None:
                        metric_parts.append(f"Acc {acc*100:.0f}%")
                    metric = ", ".join(metric_parts) if metric_parts else "-"
                    think_flag = "on" if tc.enable_thinking else "off"
                    model_disp = (MODELS.get(model_name).display_name
                                  if MODELS.get(model_name) else model_name)
                    print(
                        f"[{n}/{total}] {status:7s} {model_disp:24s} "
                        f"{tc.id:10s} think={think_flag:3s} "
                        f"time={elapsed}s  {metric}"
                    )
                return r

            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [
                    pool.submit(_task, idx, model_name, tc)
                    for idx, (model_name, tc) in enumerate(exec_plan)
                ]
                # as_completed 主要是等待 + 传播异常；_task 内部已处理所有异常
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"[ERROR] 任务线程异常：{type(e).__name__}: {e}")

        wall_elapsed = time.time() - wall_start

        # 生成输出（保持 exec_plan 原始顺序便于对比）
        self._sort_results_by_plan(exec_plan)
        self._generate_report(model_names)
        self._save_json_results()

        # 汇总成功/失败/超时
        ok = sum(1 for r in self.results if r.get("status") == "OK")
        timeout_cnt = sum(1 for r in self.results if r.get("status") == "TIMEOUT")
        error_cnt = sum(1 for r in self.results if r.get("status") == "ERROR")
        print(f"\n{'#' * 60}")
        print(
            f"  所有测试完成！总耗时 {wall_elapsed:.1f}s | "
            f"成功 {ok} | 失败 {error_cnt} | 超时 {timeout_cnt}"
        )
        print(f"  结果保存在: {self.output_dir}")
        print(f"{'#' * 60}")

    def _sort_results_by_plan(
        self, exec_plan: list[tuple[str, TestCase]]
    ) -> None:
        """按 exec_plan 顺序重排 self.results，避免并行顺序随机"""
        key_to_order = {
            (model_name, tc.id, tc.enable_thinking): i
            for i, (model_name, tc) in enumerate(exec_plan)
        }

        def _k(r: dict) -> int:
            return key_to_order.get(
                (r["model"], r["test_id"], r["enable_thinking"]),
                10_000_000,
            )

        self.results.sort(key=_k)

    # ========================================================================
    # 评估方法
    # ========================================================================
    def _evaluate(
        self, test_case: TestCase, response: dict, silent: bool = False
    ) -> dict[str, Any]:
        """评估测试结果的准确度（名称识别 + 坐标精度）"""
        evaluation: dict[str, Any] = {"accuracy": None, "details": ""}
        # 名称识别评估
        if test_case.expected_groups:
            evaluation = self._evaluate_group_accuracy(
                response["content"], test_case.expected_groups, silent=silent
            )
        # 坐标精度评估
        if test_case.has_ground_truth:
            coord_eval = self._evaluate_coordinates(
                test_case, response["content"], silent=silent
            )
            if coord_eval is not None:
                evaluation["coordinate_eval"] = coord_eval
                # 合并描述信息
                prev = evaluation.get("details", "").strip()
                evaluation["details"] = (
                    f"{prev} | {coord_eval['details']}" if prev else coord_eval["details"]
                )
        return evaluation

    def _evaluate_coordinates(
        self, test_case: TestCase, content: str, silent: bool = False
    ) -> dict[str, Any] | None:
        """
        基于 Ground Truth 评估坐标精度

        评估指标：
        - point_in_bbox 命中率：预测中心点是否落在任一 GT bbox 内（带容差）
        - mean IoU：若模型输出 bbox（4 数字），计算与最匹配 GT 的 IoU 均值
        - gt_coverage_rate：GT 中有多少比例被至少一次命中
        """
        gt_items = get_ground_truth(test_case.id)
        if not gt_items:
            return None

        preds = self._extract_predictions(content)
        if not preds:
            return {
                "predictions_count": 0,
                "gt_count": len(gt_items),
                "point_in_bbox_hits": 0,
                "point_hit_rate": 0.0,
                "gt_coverage_rate": 0.0,
                "mean_iou": None,
                "matches": [],
                "details": "未解析到任何 <point> 坐标",
            }

        matches: list[dict] = []
        hit_count = 0
        for i, pred in enumerate(preds):
            matched_gt_idx, matched_iou = self._match_prediction_to_gt(pred, gt_items)
            if matched_gt_idx is not None:
                hit_count += 1
            matches.append(
                {
                    "pred_index": i,
                    "pred_center": pred["center"],
                    "pred_bbox": pred["bbox"],
                    "matched_gt_index": matched_gt_idx,
                    "matched_gt_label": (
                        gt_items[matched_gt_idx].label if matched_gt_idx is not None else None
                    ),
                    "iou": round(matched_iou, 4) if matched_iou is not None else None,
                }
            )

        point_hit_rate = hit_count / len(preds) if preds else 0.0
        covered_gt = {m["matched_gt_index"] for m in matches if m["matched_gt_index"] is not None}
        gt_coverage_rate = len(covered_gt) / len(gt_items)

        ious = [m["iou"] for m in matches if m["iou"] is not None]
        mean_iou = sum(ious) / len(ious) if ious else None

        details = (
            f"点命中 {hit_count}/{len(preds)} ({point_hit_rate:.1%}), "
            f"GT覆盖 {len(covered_gt)}/{len(gt_items)} ({gt_coverage_rate:.1%})"
        )
        if mean_iou is not None:
            details += f", 平均IoU {mean_iou:.3f}"

        if not silent:
            print(f"\n--- 坐标评估 ---")
            print(details)

        return {
            "predictions_count": len(preds),
            "gt_count": len(gt_items),
            "point_in_bbox_hits": hit_count,
            "point_hit_rate": round(point_hit_rate, 4),
            "gt_coverage_rate": round(gt_coverage_rate, 4),
            "mean_iou": round(mean_iou, 4) if mean_iou is not None else None,
            "matches": matches,
            "details": details,
        }

    @staticmethod
    def _match_prediction_to_gt(
        pred: dict, gt_items: list[GTItem]
    ) -> tuple[int | None, float | None]:
        """
        将一个预测匹配到最合适的 GT。

        策略：
        1. 先用 point-in-bbox (带容差) 寻找最近的 GT
        2. 若无，且预测带 bbox，用 IoU >= 阈值 判断
        返回: (matched_gt_index, iou_value or None)
        """
        cx, cy = pred["center"]
        pred_bbox = pred["bbox"]

        # 策略 1: 点在 bbox 内
        for gt_i, gt in enumerate(gt_items):
            x1, y1, x2, y2 = gt.bbox
            if (
                x1 - POINT_TOLERANCE <= cx <= x2 + POINT_TOLERANCE
                and y1 - POINT_TOLERANCE <= cy <= y2 + POINT_TOLERANCE
            ):
                iou = (
                    TestRunner._compute_iou(pred_bbox, gt.bbox) if pred_bbox else None
                )
                return gt_i, iou

        # 策略 2: IoU 阈值
        if pred_bbox:
            best_iou = 0.0
            best_idx: int | None = None
            for gt_i, gt in enumerate(gt_items):
                v = TestRunner._compute_iou(pred_bbox, gt.bbox)
                if v > best_iou:
                    best_iou = v
                    best_idx = gt_i
            if best_idx is not None and best_iou >= IOU_HIT_THRESHOLD:
                return best_idx, best_iou
            # 仍返回最好 IoU 作为诊断信息，但不视为命中
            return None, best_iou if best_iou > 0 else None

        return None, None

    @staticmethod
    def _compute_iou(
        a: tuple[int, int, int, int] | None,
        b: tuple[int, int, int, int] | None,
    ) -> float:
        """计算两个 bbox 的 IoU，bbox 格式 (x1, y1, x2, y2)"""
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _evaluate_group_accuracy(
        content: str, expected_groups: list[str], silent: bool = False
    ) -> dict[str, Any]:
        """评估群组识别准确度（基于模糊匹配）"""
        # 规范化：小写，去除 Markdown 转义反斜杠和装饰字符
        content_norm = re.sub(r"\\([|*_`])", r"\1", content.lower())
        found = []
        missed = []

        for group in expected_groups:
            group_lower = group.lower()
            clean_group = re.sub(r"[♥️«»\u200d]", "", group_lower).strip()
            if clean_group in content_norm or group_lower in content_norm:
                found.append(group)
            else:
                missed.append(group)

        total = len(expected_groups)
        accuracy = len(found) / total if total > 0 else 0

        evaluation = {
            "accuracy": round(accuracy, 4),
            "found_count": len(found),
            "total_count": total,
            "found_groups": found,
            "missed_groups": missed,
            "details": f"识别 {len(found)}/{total} 个群组 (准确率: {accuracy:.1%})",
        }

        if not silent:
            print(f"\n--- 评估结果 ---")
            print(f"准确率: {accuracy:.1%} ({len(found)}/{total})")
            if missed:
                print(f"未识别: {missed}")

        return evaluation

    # ========================================================================
    # 坐标解析与标记
    # ========================================================================
    @staticmethod
    def _extract_predictions(content: str) -> list[dict]:
        """
        提取所有坐标预测，返回结构化预测列表。

        输出项：
            {"center": (x, y), "bbox": (x1, y1, x2, y2) | None}

        兼容多种厂商官方坐标格式（统一转换到 0-1000 空间）：

        A. `<point>x y</point>`             - doubao-seed-1-6-vision（2 数字）
        B. `<point>x1 y1 x2 y2</point>`     - 部分模型把 bbox 塞进 point 壳（4 数字）
        C. `<point>(x, y)</point>` / `<point>[x, y]</point>` - Qwen3-VL 带括号
        D. `<point x1=".." y1=".." [x2=".." y2=".."] />`    - qwen3-vl-32b-instruct 自闭合
        E. `<bbox>x1 y1 x2 y2</bbox>`        - doubao-1-5-vision-pro 官方 bbox
        F. `<box>x1 y1 x2 y2</box>` / `<box>(x1,y1),(x2,y2)</box>` - Qwen3-VL bbox
        G. `<tool_call>{"name":"computer_use","arguments":{"action":"left_click",
           "coordinate":[x,y]}}</tool_call>` - GUI-Plus computer_use 工具调用

        说明：
        - 所有 prompt 都要求模型输出 0-1000 相对坐标，解析器不做缩放
        - gui-plus 的 tool_call 因 system prompt 声明 resolution=1000×1000，
          天然落在 0-1000 空间
        """
        results: list[dict] = []

        def _append_from_nums(nums: list[int]) -> None:
            if len(nums) == 2:
                results.append({"center": (nums[0], nums[1]), "bbox": None})
            elif len(nums) == 4:
                x1, y1, x2, y2 = nums
                results.append(
                    {
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        # 格式 A/B/C：<point>...</point> 或 <point>... 无闭合标签
        # （qwen3-vl-32b-thinking 观察到输出 `<point>(325, 55)` 不带 </point>）
        # 使用非贪婪匹配，终止条件：</point> / 下个 < / 换行 / 字符串末尾
        for raw in re.findall(
            r"<point>\s*([^<\n]+?)\s*(?:</point>|(?=<)|$)",
            content,
            flags=re.MULTILINE,
        ):
            _append_from_nums([int(n) for n in re.findall(r"-?\d+", raw)])

        # 格式 E：<bbox>x1 y1 x2 y2</bbox>（doubao-1-5-vision-pro 官方）
        for raw in re.findall(r"<bbox>([^<]+?)</bbox>", content, flags=re.IGNORECASE):
            _append_from_nums([int(n) for n in re.findall(r"-?\d+", raw)])

        # 格式 F：<box>...</box>（Qwen3-VL 官方 bbox）
        # 注意需排除 <box_start>/<box_end> 之类的包裹 tag：严格匹配 <box> 结束标签
        for raw in re.findall(r"<box>([^<]+?)</box>", content, flags=re.IGNORECASE):
            _append_from_nums([int(n) for n in re.findall(r"-?\d+", raw)])

        # 格式 D：<point x1=".." y1=".." [x2=".." y2=".."] /> 自闭合属性（qwen3-vl-32b-instruct）
        attr_tag_pattern = r"<point\b([^>]*?)/?\s*>"
        for attrs in re.findall(attr_tag_pattern, content, flags=re.IGNORECASE):
            if "=" not in attrs:
                continue
            val_pairs = dict(
                (k.lower(), int(v))
                for k, v in re.findall(r'(\w+)\s*=\s*"?(-?\d+)"?', attrs)
            )
            x = val_pairs.get("x1") or val_pairs.get("x")
            y = val_pairs.get("y1") or val_pairs.get("y")
            x2 = val_pairs.get("x2")
            y2 = val_pairs.get("y2")
            if x is not None and y is not None:
                if x2 is not None and y2 is not None:
                    results.append(
                        {
                            "center": ((x + x2) // 2, (y + y2) // 2),
                            "bbox": (x, y, x2, y2),
                        }
                    )
                else:
                    results.append({"center": (x, y), "bbox": None})

        # 格式 G：<tool_call>{...}</tool_call>
        results.extend(TestRunner._extract_tool_call_predictions(content))

        # 兜底格式 H：裸坐标输出（无任何 tag 包装）
        # 场景：qwen3.6-flash thinking 模式偶发直接输出 `(18, 153)` / `18, 153` / `[18, 153]`
        # 极严格：仅当 content 整体 strip 后**完全匹配**纯坐标模式才采纳，避免误匹配推理文本
        if not results:
            stripped = content.strip()
            bare_match = re.fullmatch(
                r"[\(\[]?\s*(-?\d+)\s*[,，\s]\s*(-?\d+)\s*[\)\]]?",
                stripped,
            )
            if bare_match:
                x, y = int(bare_match.group(1)), int(bare_match.group(2))
                results.append({"center": (x, y), "bbox": None})

        return results

    @staticmethod
    def _extract_tool_call_predictions(content: str) -> list[dict]:
        """
        解析 GUI-Plus 的 tool_call 输出，提取 coordinate 字段。

        仅接受会携带坐标的 action 类型（click / move / drag 等）。
        因 system prompt 声明 resolution 1000×1000，坐标可直接当作 0-1000 使用。
        """
        # 需要提取坐标的动作白名单
        COORDINATE_ACTIONS = {
            "click",
            "left_click",
            "right_click",
            "middle_click",
            "double_click",
            "triple_click",
            "mouse_move",
            "left_click_drag",
        }

        results: list[dict] = []
        tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        for raw_json in re.findall(tool_call_pattern, content, flags=re.DOTALL):
            try:
                import json as _json  # 局部导入，避免上方已有冲突
                call_obj = _json.loads(raw_json)
            except _json.JSONDecodeError:
                # 松弛解析：尝试 Python 字面量（单引号等）
                try:
                    import ast
                    call_obj = ast.literal_eval(raw_json)
                except Exception:
                    continue

            if not isinstance(call_obj, dict):
                continue
            args = call_obj.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            action = args.get("action")
            coord = args.get("coordinate")
            if action not in COORDINATE_ACTIONS:
                continue
            if (
                not isinstance(coord, (list, tuple))
                or len(coord) != 2
                or not all(isinstance(v, (int, float)) for v in coord)
            ):
                continue
            results.append(
                {"center": (int(coord[0]), int(coord[1])), "bbox": None}
            )

        return results

    @classmethod
    def _extract_coordinates(cls, content: str) -> list[tuple[int, int]]:
        """向后兼容：仅返回中心点列表，供 _mark_coordinates 使用"""
        return [p["center"] for p in cls._extract_predictions(content)]

    def _mark_coordinates(
        self,
        image_path: Path,
        coordinates: list[tuple[int, int]],
        test_id: str,
        model_name: str,
    ) -> Path:
        """在图片上标记模型返回的坐标位置"""
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        img_w, img_h = image.size

        MARKER_RADIUS = 8
        MARKER_COLOR = "red"
        MARKER_WIDTH = 3

        for i, (rx, ry) in enumerate(coordinates):
            x = int(rx / COORDINATE_SCALE * img_w)
            y = int(ry / COORDINATE_SCALE * img_h)

            draw.ellipse(
                (
                    x - MARKER_RADIUS,
                    y - MARKER_RADIUS,
                    x + MARKER_RADIUS,
                    y + MARKER_RADIUS,
                ),
                outline=MARKER_COLOR,
                width=MARKER_WIDTH,
            )
            draw.text(
                (x + MARKER_RADIUS + 2, y - MARKER_RADIUS),
                str(i + 1),
                fill=MARKER_COLOR,
            )

        # 文件名带模型短标识，避免多模型间覆盖
        model_short = MODELS.get(model_name, None)
        model_tag = model_short.display_name if model_short else model_name
        safe_tag = re.sub(r"[^A-Za-z0-9_-]", "_", model_tag)
        output_path = self.output_dir / f"{test_id}_{safe_tag}_marked.png"
        image.save(output_path)
        print(f"标记图片已保存: {output_path}")
        return output_path

    # ========================================================================
    # 报告生成
    # ========================================================================
    def _generate_report(self, model_names: list[str]):
        """生成 Markdown 格式的测试报告"""
        report_path = self.output_dir / f"report_{self.run_id}.md"

        lines = [
            f"# GUI 模型性能测试报告",
            "",
            f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**待测模型**: {', '.join(model_names)}",
            f"**总执行数**: {len(self.results)}",
            "",
        ]

        # 跨模型对比摘要
        lines.extend(self._generate_cross_model_section(model_names))

        # 按模型分组的详细摘要表
        lines.extend(["## 测试结果摘要（按模型分组）", ""])
        for model_name in model_names:
            lines.extend(self._generate_model_summary(model_name))

        # 单个模型的深度思考对比
        lines.extend(self._generate_thinking_comparison(model_names))

        # 详细结果
        lines.extend(["", "## 详细结果", ""])
        for r in self.results:
            lines.extend(self._format_single_result(r))

        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n测试报告已生成: {report_path}")

    def _generate_cross_model_section(
        self, model_names: list[str]
    ) -> list[str]:
        """生成跨模型对比表（同用例不同模型）"""
        lines = ["## 跨模型对比（同用例）", ""]

        # 收集所有出现过的 test_id（保持用例顺序）
        test_ids: list[str] = []
        for r in self.results:
            if r["test_id"] not in test_ids:
                test_ids.append(r["test_id"])

        # 表头：用例 + 每个模型的(时间/名称准确率/坐标命中率)
        header = ["| 测试ID | 测试名称 | 深度思考 |"]
        for m in model_names:
            disp = MODELS[m].display_name if m in MODELS else m
            header.append(f" {disp}(时间) | {disp}(名称) | {disp}(坐标) |")
        lines.append("".join(header))

        divider = ["|--------|----------|----------|"]
        for _ in model_names:
            divider.append("-----------|-----------|-----------|")
        lines.append("".join(divider))

        # 按 (test_id, enable_thinking) 分组
        result_map: dict[tuple[str, bool], dict[str, dict]] = {}
        for r in self.results:
            key = (r["test_id"], r["enable_thinking"])
            result_map.setdefault(key, {})[r["model"]] = r

        # 输出行
        seen: set[tuple[str, bool]] = set()
        for r in self.results:
            key = (r["test_id"], r["enable_thinking"])
            if key in seen:
                continue
            seen.add(key)

            thinking_str = "是" if r["enable_thinking"] else "否"
            row = [f"| {r['test_id']} | {r['test_name']} | {thinking_str} |"]
            for m in model_names:
                mr = result_map[key].get(m)
                if mr is None:
                    row.append(" - | - | - |")
                else:
                    acc = mr.get("evaluation", {}).get("accuracy")
                    acc_str = f"{acc:.1%}" if acc is not None else "-"
                    coord = mr.get("evaluation", {}).get("coordinate_eval")
                    if coord:
                        coord_str = f"{coord['point_hit_rate']:.0%}"
                        if coord.get("mean_iou") is not None:
                            coord_str += f" (IoU {coord['mean_iou']:.2f})"
                    else:
                        coord_str = "-"
                    row.append(f" {mr['response_time']}s | {acc_str} | {coord_str} |")
            lines.append("".join(row))

        lines.append("")
        return lines

    def _generate_model_summary(self, model_name: str) -> list[str]:
        """生成单个模型的结果摘要"""
        disp = MODELS[model_name].display_name if model_name in MODELS else model_name
        lines = [f"### 模型: {disp} (`{model_name}`)", ""]

        model_results = [r for r in self.results if r["model"] == model_name]
        if not model_results:
            lines.extend(["（无结果）", ""])
            return lines

        total_time = sum(r["response_time"] for r in model_results)
        avg_time = total_time / len(model_results) if model_results else 0

        lines.extend(
            [
                f"- **用例数**: {len(model_results)}",
                f"- **总耗时**: {total_time:.2f}s",
                f"- **平均响应时间**: {avg_time:.2f}s",
                "",
                "| 测试ID | 测试名称 | 类别 | 深度思考 | 响应时间(s) | 准确率 | 状态 |",
                "|--------|----------|------|----------|-------------|--------|------|",
            ]
        )

        for r in model_results:
            acc = r.get("evaluation", {}).get("accuracy")
            acc_str = f"{acc:.1%}" if acc is not None else "-"
            thinking_str = "是" if r["enable_thinking"] else "否"
            lines.append(
                f"| {r['test_id']} | {r['test_name']} | {r['category']} "
                f"| {thinking_str} | {r['response_time']} | {acc_str} | {r['status']} |"
            )

        lines.append("")
        return lines

    def _generate_thinking_comparison(
        self, model_names: list[str]
    ) -> list[str]:
        """生成各模型内部的深度思考对比"""
        lines = ["## 深度思考 vs 非深度思考 对比", ""]

        # 配对：(非深度思考ID, 深度思考ID, 名称)
        COMPARISON_PAIRS = [
            ("RSN_001", "RSN_002", "群组信息提取"),
            ("RSN_003", "RSN_004", "界面布局分析"),
        ]

        for model_name in model_names:
            cfg = MODELS.get(model_name)
            if not cfg or not cfg.supports_thinking:
                continue  # 不支持的模型跳过

            results_by_id = {
                r["test_id"]: r for r in self.results if r["model"] == model_name
            }

            lines.append(f"### 模型: {cfg.display_name}")
            lines.append("")

            for no_think_id, think_id, name in COMPARISON_PAIRS:
                r_no = results_by_id.get(no_think_id)
                r_yes = results_by_id.get(think_id)
                if not r_no or not r_yes:
                    continue

                lines.extend(
                    [
                        f"**{name}**",
                        "",
                        "| 指标 | 非深度思考 | 深度思考 |",
                        "|------|-----------|---------|",
                        f"| 响应时间 | {r_no['response_time']}s | {r_yes['response_time']}s |",
                    ]
                )
                acc_no = r_no.get("evaluation", {}).get("accuracy")
                acc_yes = r_yes.get("evaluation", {}).get("accuracy")
                acc_no_str = f"{acc_no:.1%}" if acc_no is not None else "-"
                acc_yes_str = f"{acc_yes:.1%}" if acc_yes is not None else "-"
                lines.append(f"| 准确率 | {acc_no_str} | {acc_yes_str} |")

                tokens_no = r_no.get("usage", {}).get("total_tokens", "-")
                tokens_yes = r_yes.get("usage", {}).get("total_tokens", "-")
                lines.append(f"| 总Token | {tokens_no} | {tokens_yes} |")
                lines.append("")

        return lines

    def _format_single_result(self, r: dict) -> list[str]:
        """格式化单条测试结果为 Markdown"""
        model_disp = (
            MODELS[r["model"]].display_name if r["model"] in MODELS else r["model"]
        )
        lines = [
            f"### [{model_disp}] {r['test_id']} - {r['test_name']}",
            "",
            f"- **模型**: `{r['model']}`",
            f"- **类别**: {r['category']}",
            f"- **深度思考**: {'是' if r['enable_thinking'] else '否'}",
            f"- **响应时间**: {r['response_time']}s",
            f"- **状态**: {r['status']}",
        ]

        if r.get("usage"):
            lines.append(f"- **Token用量**: {r['usage']}")

        eval_info = r.get("evaluation", {})
        if eval_info.get("details"):
            lines.append(f"- **评估**: {eval_info['details']}")

        # 坐标评估详情
        coord_eval = eval_info.get("coordinate_eval")
        if coord_eval:
            lines.extend(
                [
                    "",
                    "**坐标评估**:",
                    "",
                    "| 指标 | 值 |",
                    "|------|-----|",
                    f"| 预测数 | {coord_eval['predictions_count']} |",
                    f"| GT 目标数 | {coord_eval['gt_count']} |",
                    (
                        f"| 点命中率 | {coord_eval['point_hit_rate']:.1%} "
                        f"({coord_eval['point_in_bbox_hits']}/{coord_eval['predictions_count']}) |"
                    ),
                    f"| GT 覆盖率 | {coord_eval['gt_coverage_rate']:.1%} |",
                    (
                        f"| 平均 IoU | {coord_eval['mean_iou']:.3f} |"
                        if coord_eval.get("mean_iou") is not None
                        else "| 平均 IoU | N/A（模型仅返回点，非 bbox）|"
                    ),
                ]
            )

        MAX_THINKING_DISPLAY = 1000
        if r.get("thinking_content"):
            lines.extend(["", "**思考过程**:", "```"])
            thinking = r["thinking_content"]
            if len(thinking) > MAX_THINKING_DISPLAY:
                lines.append(thinking[:MAX_THINKING_DISPLAY] + "\n...(内容已截断)")
            else:
                lines.append(thinking)
            lines.append("```")

        lines.extend(["", "**模型回复**:", "```", r.get("content", ""), "```"])

        if r.get("marked_image"):
            lines.extend(["", f"**标记图片**: {r['marked_image']}"])

        lines.extend(["", "---", ""])
        return lines

    def _save_json_results(self):
        """保存 JSON 格式的测试结果"""
        json_path = self.output_dir / f"results_{self.run_id}.json"
        json_path.write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON结果已保存: {json_path}")

    # ========================================================================
    # 辅助打印方法
    # ========================================================================
    @staticmethod
    def _print_test_header(test_case: TestCase, model_name: str):
        """打印测试用例头部信息"""
        print(f"\n{'=' * 60}")
        print(f"模型: {model_name}")
        print(f"测试 [{test_case.id}] {test_case.name}")
        print(f"类别: {test_case.category.value}")
        print(f"深度思考(请求): {'是' if test_case.enable_thinking else '否'}")
        print(f"描述: {test_case.description}")
        print(f"{'=' * 60}")

    @staticmethod
    def _print_response_summary(response: dict):
        """打印响应摘要"""
        print(f"\n响应时间: {response['response_time']}s")
        print(f"Token用量: {response['usage']}")
        print(f"实际启用 thinking: {response.get('thinking_applied', False)}")

        MAX_PREVIEW_LENGTH = 500

        if response["thinking_content"]:
            print(f"\n--- 思考过程 ---")
            thinking = response["thinking_content"]
            if len(thinking) > MAX_PREVIEW_LENGTH:
                print(thinking[:MAX_PREVIEW_LENGTH] + "\n...(截断)")
            else:
                print(thinking)

        print(f"\n--- 模型回复 ---")
        print(response["content"])

    @staticmethod
    def _build_error_result(
        test_case: TestCase,
        model_name: str,
        error_msg: str,
        status: str = "ERROR",
    ) -> dict[str, Any]:
        """构建错误结果。status ∈ {ERROR, TIMEOUT}"""
        return {
            "model": model_name,
            "test_id": test_case.id,
            "test_name": test_case.name,
            "category": test_case.category.value,
            "enable_thinking": test_case.enable_thinking,
            "thinking_applied": False,
            "status": status,
            "error": error_msg,
            "response_time": 0,
            "usage": {},
            "content": "",
            "thinking_content": "",
            "evaluation": {"accuracy": None, "details": f"{status}: {error_msg}"},
            "marked_image": None,
        }

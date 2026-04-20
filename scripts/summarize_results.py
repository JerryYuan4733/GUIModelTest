"""
汇总最近一次完整测试的关键指标，避免 Windows 终端中文乱码问题。

用法：
    uv run python scripts/summarize_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODELS, OUTPUT_DIR


def _latest_results() -> Path:
    candidates = sorted(OUTPUT_DIR.glob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"未找到结果文件于 {OUTPUT_DIR}")
    return candidates[-1]


def _fmt_pct(v) -> str:
    return f"{v*100:.0f}%" if isinstance(v, (int, float)) else "-"


def _display(model_name: str) -> str:
    cfg = MODELS.get(model_name)
    return cfg.display_name if cfg else model_name


def main() -> int:
    result_file = _latest_results()
    data = json.loads(result_file.read_text(encoding="utf-8"))
    print(f"# 结果文件: {result_file.name}\n")
    print(f"总条数: {len(data)}\n")

    # 按 (test_id, thinking) 分组
    groups: dict[tuple[str, bool], list[dict]] = {}
    for r in data:
        key = (r["test_id"], r["enable_thinking"])
        groups.setdefault(key, []).append(r)

    # 表头：按结果文件里实际出现的模型动态构建（保持 MODELS 注册顺序）
    present = {r["model"] for r in data}
    model_order = [m for m in MODELS.keys() if m in present]
    disp = [_display(m) for m in model_order]

    # === 表 1：定位类（POS_*）点命中率 / GT 覆盖率 ===
    print("## 定位类测试 · 点命中率 / GT 覆盖率（thinking=off / on）\n")
    header = "| 测试 | 思考 |" + "".join(f" {d} |" for d in disp)
    sep = "|------|------|" + "".join(" :---: |" for _ in disp)
    print(header)
    print(sep)

    pos_keys = sorted(
        [k for k in groups if k[0].startswith("POS_")],
        key=lambda x: (x[0], x[1]),
    )
    for (tid, thinking) in pos_keys:
        thinking_str = "on" if thinking else "off"
        row = [f"| {tid} | {thinking_str} |"]
        by_model = {r["model"]: r for r in groups[(tid, thinking)]}
        for m in model_order:
            r = by_model.get(m)
            if r is None:
                row.append(" - |")
                continue
            coord = r.get("evaluation", {}).get("coordinate_eval")
            if coord is None:
                row.append(" - |")
                continue
            hit = coord.get("point_hit_rate", 0)
            cov = coord.get("gt_coverage_rate", 0)
            t = r.get("response_time", 0)
            row.append(f" {_fmt_pct(hit)}/{_fmt_pct(cov)} ({t}s) |")
        print("".join(row))

    # === 表 2：POS 类名称识别准确率（仅 POS_002 有）===
    print("\n## 定位类测试 · 名称识别准确率（仅 POS_002）\n")
    print("| 思考 |" + "".join(f" {d} |" for d in disp))
    print("|------|" + "".join(" :---: |" for _ in disp))
    for thinking in (False, True):
        thinking_str = "on" if thinking else "off"
        row = [f"| {thinking_str} |"]
        key = ("POS_002", thinking)
        by_model = {r["model"]: r for r in groups.get(key, [])}
        for m in model_order:
            r = by_model.get(m)
            if r is None:
                row.append(" - |")
                continue
            acc = r.get("evaluation", {}).get("accuracy")
            row.append(f" {_fmt_pct(acc)} |")
        print("".join(row))

    # === 表 3：推理类（RSN_*）名称识别准确率 + 耗时 ===
    print("\n## 推理类测试 · 名称识别准确率 / 耗时\n")
    print("| 测试 | 思考 |" + "".join(f" {d} |" for d in disp))
    print("|------|------|" + "".join(" :---: |" for _ in disp))
    rsn_keys = sorted(
        [k for k in groups if k[0].startswith("RSN_")],
        key=lambda x: (x[0], x[1]),
    )
    for (tid, thinking) in rsn_keys:
        thinking_str = "on" if thinking else "off"
        row = [f"| {tid} | {thinking_str} |"]
        by_model = {r["model"]: r for r in groups[(tid, thinking)]}
        for m in model_order:
            r = by_model.get(m)
            if r is None:
                row.append(" - |")
                continue
            acc = r.get("evaluation", {}).get("accuracy")
            t = r.get("response_time", 0)
            acc_s = _fmt_pct(acc) if acc is not None else "-"
            row.append(f" {acc_s} ({t}s) |")
        print("".join(row))

    # === 表 4：模型平均响应时间 ===
    print("\n## 各模型平均响应时间（秒）\n")
    print("| 模型 | 平均耗时 | 调用数 |")
    print("|------|---------|-------|")
    for m in model_order:
        rs = [r for r in data if r["model"] == m]
        if not rs:
            print(f"| {_display(m)} | - | 0 |")
            continue
        avg = sum(r.get("response_time", 0) for r in rs) / len(rs)
        print(f"| {_display(m)} | {avg:.2f}s | {len(rs)} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())

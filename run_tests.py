"""
GUI 模型性能测试入口

支持对多个豆包视觉模型进行对比测试，包括：
- doubao-seed-1-6-vision-250815 (支持深度思考)
- doubao-1-5-vision-pro-32k-250115 (不支持深度思考)

使用方法:
    # 运行所有模型 × 所有测试
    uv run python run_tests.py

    # 只跑定位测试
    uv run python run_tests.py --category positioning

    # 只跑 1-5-vision-pro 模型
    uv run python run_tests.py --models doubao-1-5-vision-pro-32k-250115

    # 跑两个模型的指定用例
    uv run python run_tests.py --ids POS_001 POS_002

    # 包含全屏截图测试
    uv run python run_tests.py --screenshot
"""
import argparse
import sys

from src.config import DEFAULT_MODELS, MODELS
from src.test_cases import (
    ALL_TEST_CASES,
    TestCategory,
    create_screenshot_test_case,
)
from src.test_runner import TestRunner


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Doubao vision model performance test"
    )
    parser.add_argument(
        "--category",
        choices=["positioning", "reasoning", "all"],
        default="all",
        help="test category filter (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODELS.keys()),
        help=f"models to test, default: all registered models",
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="include screenshot test (requires GUI environment)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="specific test case IDs to run, e.g. POS_001 RSN_001",
    )
    parser.add_argument(
        "--thinking-matrix",
        action="store_true",
        help=(
            "对支持 thinking 的模型，把每个用例额外复制一份 enable_thinking=True"
            "（用于新加的 qwen3-vl-flash / gui-plus 等模型的 thinking 模式对比）"
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="并发任务数（默认 4，设 1 则串行）",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=120.0,
        help="单个任务超时秒数，超时自动标记为 TIMEOUT 失败（默认 120）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 按 ID 筛选
    if args.ids:
        id_set = set(args.ids)
        test_cases = [tc for tc in ALL_TEST_CASES if tc.id in id_set]
    elif args.category == "positioning":
        test_cases = [
            tc for tc in ALL_TEST_CASES if tc.category == TestCategory.POSITIONING
        ]
    elif args.category == "reasoning":
        test_cases = [
            tc for tc in ALL_TEST_CASES if tc.category == TestCategory.REASONING
        ]
    else:
        test_cases = list(ALL_TEST_CASES)

    # 可选：截图测试
    if args.screenshot:
        screenshot_tc = create_screenshot_test_case()
        if screenshot_tc:
            test_cases.append(screenshot_tc)

    if not test_cases:
        print("没有匹配的测试用例")
        sys.exit(1)

    # 模型列表
    model_names = args.models if args.models else list(DEFAULT_MODELS)

    # 执行测试
    runner = TestRunner()
    runner.run_all(
        test_cases=test_cases,
        model_names=model_names,
        thinking_matrix=args.thinking_matrix,
        parallel=args.parallel,
        task_timeout=args.task_timeout,
    )


if __name__ == "__main__":
    main()

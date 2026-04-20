"""
Ground Truth 可视化脚本

在测试图片上叠加所有 GT bbox，便于人工校准 ground_truth.py 中的坐标。

用法：
    uv run python scripts/visualize_gt.py [--test POS_002]
输出：
    output/gt_visualization_<test_id>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许以脚本方式直接运行
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw

from src.config import COORDINATE_SCALE, OUTPUT_DIR, TEST_IMAGE_PATH
from src.ground_truth import GROUND_TRUTH, GTItem


# 视觉参数
BOX_COLOR = "lime"
BOX_WIDTH = 2
LABEL_COLOR = "yellow"
LABEL_BG = (0, 0, 0, 160)


def draw_gt_on_image(
    image_path: Path,
    items: list[GTItem],
    output_path: Path,
) -> None:
    """将 GT bbox 叠加到图片上输出"""
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    draw = ImageDraw.Draw(image)

    scale_x = img_w / COORDINATE_SCALE
    scale_y = img_h / COORDINATE_SCALE

    for i, item in enumerate(items):
        rx1, ry1, rx2, ry2 = item.bbox
        x1 = int(rx1 * scale_x)
        y1 = int(ry1 * scale_y)
        x2 = int(rx2 * scale_x)
        y2 = int(ry2 * scale_y)

        # 绘制 bbox
        draw.rectangle((x1, y1, x2, y2), outline=BOX_COLOR, width=BOX_WIDTH)

        # 标签
        text = f"{i + 1}. {item.label}"
        # 文本背景（简单底色提升可读性）
        try:
            bbox_text = draw.textbbox((x2 + 4, y1), text)
            draw.rectangle(bbox_text, fill=(0, 0, 0))
        except Exception:
            pass
        draw.text((x2 + 4, y1), text, fill=LABEL_COLOR)

    image.save(output_path)
    print(f"已保存可视化结果: {output_path}")
    print(f"图片尺寸: {img_w}x{img_h}")
    print(f"共标注 {len(items)} 个目标")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize ground truth bboxes")
    parser.add_argument(
        "--test",
        default=None,
        help="test_id 筛选（默认输出所有）",
    )
    args = parser.parse_args()

    targets = (
        {args.test: GROUND_TRUTH[args.test]}
        if args.test and args.test in GROUND_TRUTH
        else GROUND_TRUTH
    )

    if not targets:
        print(f"未找到 test_id={args.test} 的 GT 数据")
        print(f"可用: {list(GROUND_TRUTH.keys())}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for test_id, items in targets.items():
        out = OUTPUT_DIR / f"gt_visualization_{test_id}.png"
        draw_gt_on_image(TEST_IMAGE_PATH, items, out)

    return 0


if __name__ == "__main__":
    sys.exit(main())

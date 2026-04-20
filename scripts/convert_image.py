"""
将 PNG 测试图片转换为 JPEG 格式并可选缩放，以加速模型推理。

使用方法:
    uv run python scripts/convert_image.py
    uv run python scripts/convert_image.py --quality 85 --max-width 1280
"""
import argparse
import os
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "resource" / "image.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "resource" / "image.jpg"


def convert_to_jpeg(
    input_path: Path,
    output_path: Path,
    quality: int = 85,
    max_width: int | None = None,
) -> None:
    """将 PNG 图片转换为 JPEG 并可选缩放"""
    img = Image.open(input_path)
    original_size = img.size
    original_bytes = os.path.getsize(input_path)

    # 转 RGB（JPEG 不支持 alpha 通道）
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 按最大宽度按比例缩放
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # 保存 JPEG
    img.save(output_path, format="JPEG", quality=quality, optimize=True)

    new_bytes = os.path.getsize(output_path)
    reduction = (1 - new_bytes / original_bytes) * 100

    print(f"原图: {input_path.name}")
    print(f"  尺寸: {original_size[0]}x{original_size[1]}")
    print(f"  大小: {original_bytes / 1024:.1f} KB")
    print(f"输出: {output_path.name}")
    print(f"  尺寸: {img.size[0]}x{img.size[1]}")
    print(f"  大小: {new_bytes / 1024:.1f} KB")
    print(f"  质量: JPEG Q{quality}")
    print(f"体积减少: {reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Convert PNG to JPEG for faster model inference")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="input image path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output image path")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100, default 85)")
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="resize to max width if larger (default: no resize)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误: 输入文件不存在 {args.input}")
        sys.exit(1)

    convert_to_jpeg(args.input, args.output, args.quality, args.max_width)


if __name__ == "__main__":
    main()

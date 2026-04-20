"""
坐标输出格式模板与解析

统一管理各厂商官方推荐的坐标输出格式。Prompt 中用占位符 `{COORD_FMT}`
表达"格式说明段落"，实际下发时由 `build_prompt_for_model` 按模型
`output_format` 填入对应文本。

各格式对应的官方文档：
- point（<point>x y</point> 0-1000）
    字节方舟 doubao-seed-1-6-vision 官方 grounding 格式
    https://www.volcengine.com/docs/82379/1616136
- bbox（<bbox>x1 y1 x2 y2</bbox> 0-1000）
    字节方舟 doubao-1-5-vision-pro 官方 grounding 格式
    https://www.volcengine.com/docs/82379/1616136
- qwen_point（<point>(x, y)</point> / <box>(x1,y1),(x2,y2)</box> 0-1000）
    阿里 Qwen3-VL Cookbook 推荐（相对坐标归一化 0-1000）
    https://developer.aliyun.com/article/1685124
- tool_call（computer_use JSON）
    阿里 gui-plus / Anthropic computer_use 标准，坐标空间由 system prompt
    中声明的 resolution 决定（本项目固定为 1000×1000）
"""
from __future__ import annotations

# Prompt 占位符：TestCase.prompt 中以此 token 表示"格式说明段"
COORD_FMT_PLACEHOLDER = "{COORD_FMT}"

# 单目标（定位单个元素，返回一个 point / bbox）
SINGLE_TARGET_INSTRUCTIONS: dict[str, str] = {
    "point": (
        "请以 <point>x y</point> 格式返回该位置的中心坐标，"
        "坐标范围为 0-1000（图像宽高均归一化到此区间）。\n"
        "只需要返回坐标，不需要其他信息。"
    ),
    "bbox": (
        "请以 <bbox>x1 y1 x2 y2</bbox> 格式返回该位置的边界框，"
        "坐标范围为 0-1000（图像宽高均归一化到此区间）。\n"
        "只需要返回坐标，不需要其他信息。"
    ),
    "qwen_point": (
        "请以 <point>(x, y)</point> 格式返回该位置的中心坐标，"
        "坐标范围为 0-1000（图像宽高均归一化到此区间）。\n"
        "只需要返回坐标，不需要其他信息。"
    ),
    "tool_call": (
        "请使用 computer_use 工具的 click 动作，"
        "coordinate 字段填入 [x, y] 的像素坐标。"
    ),
}

# 多目标（返回多个条目，每个条目含名称 + 坐标）
# 用于 POS_002 / POS_003 这类批量定位用例
MULTI_TARGET_INSTRUCTIONS: dict[str, str] = {
    "point": (
        "请以以下格式逐行输出所有目标（坐标指向头像/图标中心）：\n"
        "1. [元素名称] <point>x y</point>\n"
        "2. [元素名称] <point>x y</point>\n"
        "...\n"
        "坐标范围为 0-1000（图像宽高均归一化到此区间）。"
    ),
    "bbox": (
        "请以以下格式逐行输出所有目标（用边界框描述头像/图标区域）：\n"
        "1. [元素名称] <bbox>x1 y1 x2 y2</bbox>\n"
        "2. [元素名称] <bbox>x1 y1 x2 y2</bbox>\n"
        "...\n"
        "坐标范围为 0-1000（图像宽高均归一化到此区间）。"
    ),
    "qwen_point": (
        "请以以下格式逐行输出所有目标（坐标指向头像/图标中心）：\n"
        "1. [元素名称] <point>(x, y)</point>\n"
        "2. [元素名称] <point>(x, y)</point>\n"
        "...\n"
        "坐标范围为 0-1000（图像宽高均归一化到此区间）。"
    ),
    "tool_call": (
        "请对每个目标元素调用 computer_use 工具的 click 动作，"
        "coordinate 字段填入像素坐标 [x, y]。"
    ),
}


def build_prompt_for_model(
    prompt_template: str,
    output_format: str,
    multi_target: bool = False,
) -> str:
    """
    将 TestCase.prompt 中的 {COORD_FMT} 占位符替换为模型官方推荐的格式说明。

    Args:
        prompt_template: 含 {COORD_FMT} 占位符的任务描述
        output_format: 模型的 output_format（"point"/"bbox"/"qwen_point"/"tool_call"）
        multi_target: 是否为多目标用例（POS_002/POS_003 等）
    """
    if COORD_FMT_PLACEHOLDER not in prompt_template:
        # 兼容：TC 未使用占位符时原样返回（推理类用例无需坐标格式）
        return prompt_template

    table = MULTI_TARGET_INSTRUCTIONS if multi_target else SINGLE_TARGET_INSTRUCTIONS
    instruction = table.get(output_format, table["point"])
    return prompt_template.replace(COORD_FMT_PLACEHOLDER, instruction)

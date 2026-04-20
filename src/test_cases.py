"""
测试用例定义模块

定义所有用于评测 doubao-seed-1-6-vision-250815 模型的测试用例，
包含定位能力和推理分析能力两大类。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .config import TEST_IMAGE_PATH, TEST_IMAGE2_PATH, EXPECTED_GROUPS, OUTPUT_DIR


# ============================================================================
# 测试类别枚举
# ============================================================================
class TestCategory(Enum):
    """测试类别"""
    POSITIONING = "定位能力"
    REASONING = "推理分析"


# ============================================================================
# 测试用例数据类
# ============================================================================
@dataclass
class TestCase:
    """测试用例"""
    id: str
    name: str
    category: TestCategory
    prompt: str
    image_path: Path
    enable_thinking: bool = False
    expected_groups: list[str] = field(default_factory=list)
    description: str = ""
    # 若为 True，评估时从 ground_truth.py 查找与 id 同名的 GT 数据进行坐标评估
    has_ground_truth: bool = False
    # 是否为多目标用例（POS_002/POS_003 等需要批量输出多条）
    # 影响 prompt 中 {COORD_FMT} 占位符的填充：使用 MULTI_TARGET_INSTRUCTIONS
    multi_target: bool = False


# ============================================================================
# 定位能力测试用例
# ============================================================================
POSITIONING_SINGLE_RIGHT = TestCase(
    id="POS_001a",
    name="单个图标定位(右侧详情面板)",
    category=TestCategory.POSITIONING,
    prompt=(
        "请在图片 **右侧的频道详情面板顶部**，找到 'League of Legends Community' "
        "这个频道的圆形大头像的位置（紫色背景，位于订阅数上方）。\n"
        "{COORD_FMT}"
    ),
    image_path=TEST_IMAGE_PATH,
    description="测试模型对右侧详情面板头像的精确定位能力（消除左右歧义）",
    has_ground_truth=True,
)

POSITIONING_SINGLE_LEFT = TestCase(
    id="POS_001b",
    name="单个图标定位(左侧搜索列表)",
    category=TestCategory.POSITIONING,
    prompt=(
        "请在图片 **左侧的搜索结果列表** 中，找到名为 'League of Legends Community' "
        "的第一个条目（不是 'League of Legends Community Chat'）的头像位置。\n"
        "{COORD_FMT}"
    ),
    image_path=TEST_IMAGE_PATH,
    description="测试模型对左侧列表首项头像的精确定位能力（消除左右歧义）",
    has_ground_truth=True,
)

POSITIONING_GROUPS_ONLY = TestCase(
    id="POS_003",
    name="仅群组类型定位(细粒度类型过滤)",
    category=TestCategory.POSITIONING,
    prompt=(
        "请在图片 **左侧的搜索结果列表** 中，找出所有条目里 **类型为「群组」** 的条目。\n\n"
        "类型判断依据：每个条目名称前面会有一个小图标表示其类型：\n"
        "- 👥 **两个人的小头像** = 群组（target，需要返回）\n"
        "- 📢 喇叭 = 频道 (channel，忽略)\n"
        "- 🤖 机器人 = bot（忽略）\n"
        "- 💬 气泡 = 消息/帖子（忽略）\n\n"
        "请**只返回类型为「群组」**的条目，按从上到下顺序。坐标指向头像(avatar)的中心。\n\n"
        "{COORD_FMT}"
    ),
    image_path=TEST_IMAGE_PATH,
    description="测试模型是否能通过类型 icon 过滤出特定类别（仅群组），要求高 Precision + Recall",
    has_ground_truth=True,
    multi_target=True,
)

POSITIONING_MULTI = TestCase(
    id="POS_002",
    name="多个群组图标定位",
    category=TestCategory.POSITIONING,
    prompt=(
        "请在图片左侧的搜索结果列表中，找到所有群组/频道条目的图标(头像)位置。\n"
        "对每个群组，请返回其名称和图标中心坐标。请按从上到下的顺序列出。\n\n"
        "{COORD_FMT}"
    ),
    image_path=TEST_IMAGE_PATH,
    expected_groups=EXPECTED_GROUPS,
    description="测试模型对多个UI元素的批量定位能力",
    has_ground_truth=True,
    multi_target=True,
)


# ============================================================================
# 推理分析测试用例（非深度思考 vs 深度思考）
# ============================================================================
REASONING_GROUPS_NO_THINKING = TestCase(
    id="RSN_001",
    name="群组信息提取(非深度思考)",
    category=TestCategory.REASONING,
    prompt=(
        "请仔细分析这张图片，这是一个即时通讯应用的界面。\n"
        "请从左侧搜索结果列表中，识别所有可见的群组/频道。\n"
        "对每个群组，请提供以下信息：\n"
        "1. 群组名称\n"
        "2. @handle（如果可见）\n"
        "3. 群组图标的简要描述\n"
        "4. 群组图标中心的大致坐标（x, y），坐标范围 0-1000\n\n"
        "请以表格格式输出结果。"
    ),
    image_path=TEST_IMAGE_PATH,
    enable_thinking=False,
    expected_groups=EXPECTED_GROUPS,
    description="测试模型在非深度思考模式下的推理分析能力",
)

REASONING_GROUPS_WITH_THINKING = TestCase(
    id="RSN_002",
    name="群组信息提取(深度思考)",
    category=TestCategory.REASONING,
    prompt=(
        "请仔细分析这张图片，这是一个即时通讯应用的界面。\n"
        "请从左侧搜索结果列表中，识别所有可见的群组/频道。\n"
        "对每个群组，请提供以下信息：\n"
        "1. 群组名称\n"
        "2. @handle（如果可见）\n"
        "3. 群组图标的简要描述\n"
        "4. 群组图标中心的大致坐标（x, y），坐标范围 0-1000\n\n"
        "请以表格格式输出结果。"
    ),
    image_path=TEST_IMAGE_PATH,
    enable_thinking=True,
    expected_groups=EXPECTED_GROUPS,
    description="测试模型在深度思考模式下的推理分析能力",
)


# ============================================================================
# 复杂推理测试用例
# ============================================================================
COMPLEX_REASONING_NO_THINKING = TestCase(
    id="RSN_003",
    name="界面布局分析(非深度思考)",
    category=TestCategory.REASONING,
    prompt=(
        "请详细分析这张应用截图：\n"
        "1. 这是什么应用？\n"
        "2. 当前用户在执行什么操作？\n"
        "3. 界面分为哪几个区域？每个区域的功能是什么？\n"
        "4. 右侧面板显示了哪个群组的详细信息？\n"
        "5. 该群组有多少订阅者？有多少帖子？\n"
        "6. 中间区域最新的消息内容是关于什么的？\n"
    ),
    image_path=TEST_IMAGE_PATH,
    enable_thinking=False,
    description="测试模型在非深度思考模式下的复杂推理能力",
)

COMPLEX_REASONING_WITH_THINKING = TestCase(
    id="RSN_004",
    name="界面布局分析(深度思考)",
    category=TestCategory.REASONING,
    prompt=(
        "请详细分析这张应用截图：\n"
        "1. 这是什么应用？\n"
        "2. 当前用户在执行什么操作？\n"
        "3. 界面分为哪几个区域？每个区域的功能是什么？\n"
        "4. 右侧面板显示了哪个群组的详细信息？\n"
        "5. 该群组有多少订阅者？有多少帖子？\n"
        "6. 中间区域最新的消息内容是关于什么的？\n"
    ),
    image_path=TEST_IMAGE_PATH,
    enable_thinking=True,
    description="测试模型在深度思考模式下的复杂推理能力",
)


# ============================================================================
# 全屏截图测试（动态生成）
# ============================================================================
def create_screenshot_test_case() -> TestCase | None:
    """
    创建全屏截图测试用例

    截取当前屏幕并生成测试用例。需要 GUI 环境支持。
    """
    try:
        from PIL import ImageGrab

        screenshot_dir = OUTPUT_DIR / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = screenshot_dir / "screen_capture.png"

        screenshot = ImageGrab.grab()
        screenshot.save(screenshot_path)
        print(f"已截取当前屏幕: {screenshot_path} ({screenshot.size[0]}x{screenshot.size[1]})")

        return TestCase(
            id="SCR_001",
            name="全屏截图分析",
            category=TestCategory.REASONING,
            prompt=(
                "请分析这张电脑屏幕截图：\n"
                "1. 当前打开了哪些应用程序或窗口？\n"
                "2. 描述桌面的主要布局\n"
                "3. 是否有任何对话窗口或通知？\n"
                "4. 列出你能识别的所有可点击元素（按钮、图标、链接等），"
                "并给出它们的大致坐标（0-1000范围）\n"
            ),
            image_path=screenshot_path,
            enable_thinking=False,
            description="测试模型对实时屏幕截图的分析能力",
        )
    except Exception as e:
        print(f"截图失败（可能缺少GUI环境）: {e}")
        return None


# ============================================================================
# 所有预定义测试用例
# ============================================================================
POSITIONING_INPUT_BOX = TestCase(
    id="POS_004",
    name="搜索输入框定位(点击输入框)",
    category=TestCategory.POSITIONING,
    prompt=(
        "请在图片中找到 **顶部的搜索输入框（Search input / 搜索框）**，"
        "即界面最上方带有 'Search' 占位符的横向长条形输入框。\n"
        "{COORD_FMT}"
    ),
    image_path=TEST_IMAGE2_PATH,
    description="测试模型是否能正确定位一个大范围的可交互 UI 元素（搜索输入框）",
    has_ground_truth=True,
)


ALL_TEST_CASES: list[TestCase] = [
    POSITIONING_SINGLE_RIGHT,
    POSITIONING_SINGLE_LEFT,
    POSITIONING_MULTI,
    POSITIONING_GROUPS_ONLY,
    POSITIONING_INPUT_BOX,
    REASONING_GROUPS_NO_THINKING,
    REASONING_GROUPS_WITH_THINKING,
    COMPLEX_REASONING_NO_THINKING,
    COMPLEX_REASONING_WITH_THINKING,
]

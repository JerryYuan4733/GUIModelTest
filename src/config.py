"""
配置常量模块

集中管理项目配置：API参数、路径、模型参数和预期结果（Ground Truth）。
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# ============================================================================
# 项目路径配置
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
RESOURCE_DIR = PROJECT_ROOT / "resource"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 加载 .env 文件（如存在）
load_dotenv(PROJECT_ROOT / ".env")

# ============================================================================
# API 配置（按 Provider 分组）
# ============================================================================
# ARK = 火山方舟（字节跳动）
ARK_API_KEY = os.getenv("ARK_API_KEY")
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# DashScope = 阿里云百炼（OpenAI 兼容模式）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# Provider 行为差异表
# thinking_style:
#   - "ark":        extra_body={"thinking": {"type": "enabled", "budget_tokens": N}}
#   - "dashscope":  extra_body={"enable_thinking": True|False}
PROVIDER_CONFIG: dict[str, dict] = {
    "ark": {
        "base_url": ARK_BASE_URL,
        "api_key": ARK_API_KEY,
        "api_key_env": "ARK_API_KEY",
        "thinking_style": "ark",
    },
    "dashscope": {
        "base_url": DASHSCOPE_BASE_URL,
        "api_key": DASHSCOPE_API_KEY,
        "api_key_env": "DASHSCOPE_API_KEY",
        "thinking_style": "dashscope",
    },
}


# ============================================================================
# 模型注册表
# ============================================================================
@dataclass(frozen=True)
class ModelConfig:
    """模型配置"""
    name: str                           # 模型 ID
    supports_thinking: bool             # 是否支持深度思考
    display_name: str = ""              # 报告中的显示名
    provider: str = "ark"               # "ark" | "dashscope"
    # GUI-Plus 等专用模型需要强制注入官方 system prompt
    requires_system_prompt: bool = False
    # 输出坐标格式（按各厂商官方推荐）：
    #   - "point":      <point>x y</point>（0-1000 相对坐标）
    #                   doubao-seed-1-6-vision 官方格式
    #   - "bbox":       <bbox>x1 y1 x2 y2</bbox>（0-1000 相对坐标）
    #                   doubao-1-5-vision-pro 官方 grounding 格式
    #   - "qwen_point": <point>(x, y)</point> / <box>(x1,y1),(x2,y2)</box>
    #                   （Qwen3-VL Cookbook 推荐格式，带括号包裹，0-1000 相对坐标）
    #   - "tool_call":  <tool_call>{...}</tool_call>（GUI-Plus 的 computer_use JSON，1000×1000 空间）
    output_format: str = "point"
    # vl_high_resolution_images（仅 DashScope 视觉模型需要）
    vl_high_resolution: bool = False

    def __post_init__(self):
        if not self.display_name:
            object.__setattr__(self, "display_name", self.name)


# 所有待测模型
MODELS: dict[str, ModelConfig] = {
    # ------------------ 火山方舟（字节跳动）------------------
    "doubao-seed-1-6-vision-250815": ModelConfig(
        name="doubao-seed-1-6-vision-250815",
        supports_thinking=True,
        display_name="seed-1-6-vision",
        provider="ark",
    ),
    "doubao-1-5-vision-pro-32k-250115": ModelConfig(
        name="doubao-1-5-vision-pro-32k-250115",
        supports_thinking=False,
        display_name="1-5-vision-pro",
        provider="ark",
        # 1-5-vision-pro 的 grounding 官方推荐 <bbox>x1 y1 x2 y2</bbox>
        output_format="bbox",
    ),
    # ------------------ 阿里云百炼（Qwen 系列）--------------
    "qwen3-vl-flash-2026-01-22": ModelConfig(
        name="qwen3-vl-flash-2026-01-22",
        supports_thinking=True,
        display_name="qwen3-vl-flash",
        provider="dashscope",
        # Qwen3-VL Cookbook 推荐格式：<point>(x, y)</point> / <box>(x1,y1),(x2,y2)</box>
        output_format="qwen_point",
        vl_high_resolution=True,
    ),
    # Plus：更大参数，视觉理解最强的通用多模态模型
    "qwen3-vl-plus-2025-12-19": ModelConfig(
        name="qwen3-vl-plus-2025-12-19",
        supports_thinking=True,
        display_name="qwen3-vl-plus",
        provider="dashscope",
        output_format="qwen_point",
        vl_high_resolution=True,
    ),
    # 32B-Instruct：开源权重规格的非思考版本（enable_thinking 不生效，不传该参数）
    "qwen3-vl-32b-instruct": ModelConfig(
        name="qwen3-vl-32b-instruct",
        supports_thinking=False,  # 无 thinking 开关
        display_name="qwen3-vl-32b-instruct",
        provider="dashscope",
        output_format="qwen_point",
        vl_high_resolution=True,
    ),
    # 32B-Thinking：默认开启思考，DashScope 端不接受关闭
    "qwen3-vl-32b-thinking": ModelConfig(
        name="qwen3-vl-32b-thinking",
        supports_thinking=False,  # 不允许关闭，对外视作不可切换
        display_name="qwen3-vl-32b-thinking",
        provider="dashscope",
        output_format="qwen_point",
        vl_high_resolution=True,
    ),
    # Qwen3.6-Flash：Qwen3.6 原生视觉语言 Flash 模型（3.5-Flash 升级版）
    # 官方说明：重点增强 agentic coding、数学推理、代码推理、多模态识别（OCR/万物识别）
    # https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/detail/qwen3.6-flash
    "qwen3.6-flash": ModelConfig(
        name="qwen3.6-flash",
        supports_thinking=True,
        display_name="qwen3.6-flash",
        provider="dashscope",
        # 同 qwen3-vl 系列：Cookbook 推荐 <point>(x, y)</point> / <box>(x1,y1),(x2,y2)</box>
        output_format="qwen_point",
        vl_high_resolution=True,
    ),
    "gui-plus-2026-02-26": ModelConfig(
        name="gui-plus-2026-02-26",
        supports_thinking=True,
        display_name="gui-plus",
        provider="dashscope",
        requires_system_prompt=True,
        output_format="tool_call",
        vl_high_resolution=True,
    ),
}

# 默认待测模型（可通过命令行覆盖）
DEFAULT_MODELS: list[str] = list(MODELS.keys())

# ============================================================================
# 模型推理配置
# ============================================================================
TEMPERATURE = 0.0
COORDINATE_SCALE = 1000  # 模型输出坐标范围 (0-1000)
THINKING_BUDGET_TOKENS = 10000  # 深度思考 token 预算

# ============================================================================
# 测试图片路径
# ============================================================================
TEST_IMAGE_PATH = RESOURCE_DIR / "image.jpg"
# POS_004 等用例：顶部带搜索输入框的 Telegram 界面
TEST_IMAGE2_PATH = RESOURCE_DIR / "image2.png"

# ============================================================================
# 预期群组信息（Ground Truth）
# 基于 resource/image.png 中 Telegram 搜索结果的群组列表
# ============================================================================
EXPECTED_GROUPS = [
    "League of Legends Community",
    "Starky | League of Legends + IRL",
    "League of Legends",
    "League of Legends: Wild Rift",
    "League of Legends Community Chat",
    "League of Legends News",
    "League Of Legends Iran",
    "League of legend",
    "League of legends",
    "League off Legend",
    "«League of legends»",
]

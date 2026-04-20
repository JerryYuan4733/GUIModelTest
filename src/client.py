"""
多 Provider 视觉模型客户端

统一封装对 ARK（火山方舟）和 DashScope（阿里云百炼）的 Chat Completions 调用。
两家都是 OpenAI 兼容协议，通过 `openai` SDK 按不同 base_url/api_key 访问。

Provider 之间的主要差异：
- `thinking_style="ark"`      → extra_body={"thinking": {"type": "enabled", "budget_tokens": N}}
- `thinking_style="dashscope"` → extra_body={"enable_thinking": True|False}
- DashScope 视觉模型额外支持 `vl_high_resolution_images=True` 提升图像细节
- gui-plus 等模型需要强制注入官方 system prompt 才能输出 tool_call 格式
"""
import base64
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import (
    MODELS,
    PROVIDER_CONFIG,
    TEMPERATURE,
    THINKING_BUDGET_TOKENS,
    ModelConfig,
)
from .gui_plus_prompts import GUI_PLUS_SYSTEM_PROMPT


class VisionClient:
    """多 Provider 视觉模型客户端（按 Provider 缓存 OpenAI 实例）"""

    _instance = None
    # provider_name -> OpenAI client 实例
    _clients: dict[str, OpenAI] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_client(self, provider: str) -> OpenAI:
        """按 provider 懒加载并缓存 OpenAI client"""
        if provider in self._clients:
            return self._clients[provider]

        cfg = PROVIDER_CONFIG.get(provider)
        if cfg is None:
            raise ValueError(f"未知 provider: {provider}（支持: {list(PROVIDER_CONFIG.keys())}）")

        if not cfg["api_key"]:
            raise ValueError(
                f"缺少 API Key: 请设置环境变量 {cfg['api_key_env']}\n"
                f"  Windows: set {cfg['api_key_env']}=your_key\n"
                f"  Linux:   export {cfg['api_key_env']}=your_key"
            )

        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
        self._clients[provider] = client
        return client

    # ========================================================================
    # 消息构建
    # ========================================================================
    @staticmethod
    def encode_image_to_base64(image_path: str | Path) -> str:
        """将图片编码为 Base64 字符串"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def build_messages(
        self,
        prompt: str,
        image_path: str | Path,
        model_cfg: ModelConfig | None = None,
        multi_target: bool = False,
    ) -> list[dict[str, Any]]:
        """
        构建 Chat API 消息体

        - 根据 model_cfg 决定是否注入 system prompt（如 gui-plus 官方提示词）
        - 根据 model_cfg.output_format 将 prompt 中的 {COORD_FMT} 占位符
          替换为该厂商官方推荐的坐标格式说明
        """
        # 1) 按模型官方推荐格式填充 prompt 中的坐标格式说明段
        from .coord_formats import build_prompt_for_model
        output_format = model_cfg.output_format if model_cfg else "point"
        final_prompt = build_prompt_for_model(
            prompt, output_format=output_format, multi_target=multi_target
        )

        image_format = Path(image_path).suffix.lstrip(".").lower()
        if image_format == "jpg":
            image_format = "jpeg"
        base64_image = self.encode_image_to_base64(image_path)

        messages: list[dict[str, Any]] = []

        # GUI-Plus 等模型：强制注入官方 system prompt
        if model_cfg and model_cfg.requires_system_prompt:
            messages.append(
                {"role": "system", "content": GUI_PLUS_SYSTEM_PROMPT}
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": final_prompt},
                ],
            }
        )
        return messages

    # ========================================================================
    # 模型调用
    # ========================================================================
    def call(
        self,
        messages: list[dict[str, Any]],
        model_name: str,
        enable_thinking: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        调用模型推理

        Returns dict:
            - content: 模型回复
            - thinking_content: 思考过程
            - response_time: 耗时（秒）
            - usage: token 用量
            - thinking_applied: 实际是否开启 thinking
        """
        model_cfg = MODELS.get(model_name)
        if model_cfg is None:
            raise ValueError(f"未注册的模型: {model_name}")

        provider = model_cfg.provider
        client = self._get_client(provider)

        thinking_applied = bool(
            enable_thinking and model_cfg.supports_thinking
        )

        # 组装 extra_body
        extra_body: dict[str, Any] = {}
        thinking_style = PROVIDER_CONFIG[provider]["thinking_style"]

        if thinking_style == "ark":
            # ARK 统一通过 thinking 字段显式声明，即便非 thinking 也显式 disabled
            extra_body["thinking"] = (
                {"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS}
                if thinking_applied
                else {"type": "disabled"}
            )
        elif thinking_style == "dashscope":
            # DashScope 视觉/GUI 模型
            # 仅对可切换 thinking 的模型发送 enable_thinking 参数；
            # 32b-thinking（强制 on）/ 32b-instruct（强制 off）等不可切换模型省略此参数
            if model_cfg.supports_thinking:
                extra_body["enable_thinking"] = thinking_applied
            if model_cfg.vl_high_resolution:
                extra_body["vl_high_resolution_images"] = True

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": TEMPERATURE,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        # 单次请求超时（HTTP 层），超过则抛 openai.APITimeoutError
        if timeout is not None:
            kwargs["timeout"] = timeout

        # 调用并计时
        start_time = time.time()
        response = client.chat.completions.create(**kwargs)
        elapsed = time.time() - start_time

        choice = response.choices[0]
        content = choice.message.content or ""
        thinking_content = getattr(choice.message, "reasoning_content", "") or ""

        usage: dict[str, Any] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {
            "content": content,
            "thinking_content": thinking_content,
            "response_time": round(elapsed, 3),
            "usage": usage,
            "thinking_applied": thinking_applied,
        }


# 向后兼容别名：旧代码使用 ArkClient() 仍然可用
ArkClient = VisionClient

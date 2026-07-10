"""火山引擎 Ark API 公共配置模块。

使用方式:
    from examples.volcengine_config import setup_volcengine

    # 在 main() 开头调用即可
    setup_volcengine()
"""

import os
from openai import AsyncOpenAI



from agents import (
    OpenAIChatCompletionsModel,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

# 火山引擎 Ark API 配置常量
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
ARK_MODEL_NAME = "glm-5.2"

_client: AsyncOpenAI | None = None
_model: OpenAIChatCompletionsModel | None = None


def get_client() -> AsyncOpenAI:
    """获取火山引擎 OpenAI 兼容客户端。"""
    global _client
    if _client is None:
        setup_volcengine()
    assert _client is not None
    return _client


def get_model() -> OpenAIChatCompletionsModel:
    """获取火山引擎 GLM-5.2 模型实例。"""
    global _model
    if _model is None:
        setup_volcengine()
    assert _model is not None
    return _model


def setup_volcengine() -> None:
    """初始化火山引擎 Ark API 配置（幂等，可多次调用）。"""
    global _client, _model
    if _client is not None:
        return

    _client = AsyncOpenAI(base_url=ARK_BASE_URL, api_key=ARK_API_KEY)
    set_default_openai_client(client=_client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(disabled=True)

    _model = OpenAIChatCompletionsModel(model=ARK_MODEL_NAME, openai_client=_client)

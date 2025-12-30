#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：llm.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:06 
'''

import logging
from typing import Any, List, Iterator, AsyncIterator

# 【升级】引入 ChatOpenAI 和基础消息类型
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel

from ..core.interfaces import LLMInterface
from ..core.exceptions import InitializationError, GenerationError
from ..config.models import OllamaLLMConfig, OpenAILLMConfig

logger = logging.getLogger(__name__)

class OpenAILLM(LLMInterface):
    """
    使用 OpenAI 兼容协议 (SiliconFlow, DeepSeek, Local vLLM) 的 Chat Model 实现。
    """

    def __init__(self, config: OpenAILLMConfig):
        self.config = config
        try:
            self._model = ChatOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                streaming=True, # 默认开启流式能力
                request_timeout=60, # 防止云端卡死
            )
            logger.info(f"OpenAI-compatible LLM initialized. URL: {config.base_url}, Model: {config.model_name}")
        except Exception as e:
            raise InitializationError("OpenAILLM", f"Failed to init API client", e)

    def get_langchain_llm(self) -> BaseChatModel:
        return self._model

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> str:
        """同步调用"""
        try:
            response = self._model.invoke(messages, **kwargs)
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception as e:
            raise GenerationError(f"OpenAI generation failed: {str(e)}", e)

    async def ainvoke(self, messages: List[BaseMessage], **kwargs: Any) -> str:
        """【新增】异步调用 (FastAPI 核心依赖)"""
        try:
            # LangChain 的 ChatModel 原生支持 ainvoke
            response = await self._model.ainvoke(messages, **kwargs)
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception as e:
            raise GenerationError(f"OpenAI async generation failed: {str(e)}", e)

    def stream(self, messages: List[BaseMessage], **kwargs: Any) -> Iterator[str]:
        """同步流式生成"""
        try:
            for chunk in self._model.stream(messages, **kwargs):
                yield chunk.content
        except Exception as e:
            raise GenerationError(f"Stream generation failed: {str(e)}", e)

    async def astream(self, messages: List[BaseMessage], **kwargs: Any) -> AsyncIterator[str]:
        """【新增】异步流式生成 (SSE 核心依赖)"""
        try:
            async for chunk in self._model.astream(messages, **kwargs):
                yield chunk.content
        except Exception as e:
            raise GenerationError(f"Async stream generation failed: {str(e)}", e)

# 暂时保留 Ollama 类占位，如果暂不使用可简化，
# 但为了保持工厂模式兼容性，建议保留或同步升级。
class OllamaLLM(LLMInterface):
    """Ollama 实现 (占位/简化版)"""
    def __init__(self, config: OllamaLLMConfig):
        from langchain_community.chat_models import ChatOllama
        self._model = ChatOllama(
            base_url=config.base_url,
            model=config.model_name,
            temperature=config.temperature,
            num_ctx=config.num_ctx
        )

    def get_langchain_llm(self):
        return self._model

    def invoke(self, messages, **kwargs):
        return self._model.invoke(messages, **kwargs).content

    async def ainvoke(self, messages, **kwargs):
        res = await self._model.ainvoke(messages, **kwargs)
        return res.content

    def stream(self, messages, **kwargs):
        for c in self._model.stream(messages, **kwargs):
            yield c.content

    async def astream(self, messages, **kwargs):
        async for c in self._model.astream(messages, **kwargs):
            yield c.content
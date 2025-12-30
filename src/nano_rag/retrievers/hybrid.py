#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：hybrid.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:04 
'''

import logging
from typing import List

# 【修改点】直接从本地模块导入，彻底解决环境报错
from .ensemble_local import EnsembleRetriever

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..core.interfaces import RetrieverInterface
from ..core.exceptions import InitializationError, RetrievalError
from ..config.models import HybridRetrieverConfig

logger = logging.getLogger(__name__)


class HybridRetriever(RetrieverInterface):
    """
    混合检索器实现。
    """

    def __init__(self, config: HybridRetrieverConfig, sub_retrievers: List[RetrieverInterface], weights: List[float]):
        self.config = config

        if len(sub_retrievers) != len(weights) or len(sub_retrievers) < 2:
            raise InitializationError("HybridRetriever",
                                      "Number of sub-retrievers and weights must match and be at least 2.")

        langchain_retrievers = [r.get_langchain_retriever() for r in sub_retrievers]

        try:
            # 这里使用的就是我们本地定义的 EnsembleRetriever 了
            self._retriever = EnsembleRetriever(retrievers=langchain_retrievers, weights=weights)
            logger.info(f"HybridRetriever initialized (Local Implementation). Weights: {weights}")
        except Exception as e:
            raise InitializationError("HybridRetriever", "Failed to create EnsembleRetriever", e)

    def get_langchain_retriever(self) -> BaseRetriever:
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        try:
            results = self._retriever.invoke(query)
            return results[:self.config.top_k]
        except Exception as e:
            raise RetrievalError(f"Error during Hybrid retrieval for query: {query}", e)

    async def aretrieve(self, query: str) -> List[Document]:
        """【新增】异步混合检索"""
        try:
            # 调用本地实现的 EnsembleRetriever 的 ainvoke
            results = await self._retriever.ainvoke(query)
            return results[:self.config.top_k]
        except Exception as e:
            raise RetrievalError(f"Error during Hybrid async retrieval for query: {query}", e)
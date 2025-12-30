#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：vector.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:04 
'''

import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..core.interfaces import RetrieverInterface, VectorStoreInterface
from ..core.exceptions import InitializationError, RetrievalError
from ..config.models import VectorRetrieverConfig

logger = logging.getLogger(__name__)


class VectorRetriever(RetrieverInterface):
    """基于向量存储的检索器实现 (Async Ready)。"""

    def __init__(self, config: VectorRetrieverConfig, vector_store: VectorStoreInterface):
        self.config = config
        if not vector_store.is_initialized:
            raise InitializationError("VectorRetriever", "VectorStore is not initialized.")

        search_kwargs = {"k": self.config.top_k}
        if self.config.search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = self.config.score_threshold

        try:
            self._retriever = vector_store.as_retriever(
                search_type=self.config.search_type,
                search_kwargs=search_kwargs
            )
            logger.info(f"VectorRetriever initialized.")
        except Exception as e:
            raise InitializationError("VectorRetriever", "Failed to create retriever", e)

    def get_langchain_retriever(self) -> BaseRetriever:
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        try:
            return self._retriever.invoke(query)
        except Exception as e:
            raise RetrievalError(f"Vector retrieval error: {query}", e)

    async def aretrieve(self, query: str) -> List[Document]:
        """【新增】异步检索"""
        try:
            # LangChain Retriever 原生支持 ainvoke
            return await self._retriever.ainvoke(query)
        except Exception as e:
            raise RetrievalError(f"Async vector retrieval error: {query}", e)
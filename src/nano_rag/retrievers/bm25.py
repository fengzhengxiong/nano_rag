#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：bm25.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:03 
'''

import logging
import pickle
from pathlib import Path
from typing import List, Optional

# 【修复点 1】给 LangChain 的类起别名
from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..core.interfaces import RetrieverInterface
from ..core.exceptions import InitializationError, DataProcessingError, RetrievalError
from ..config.models import BM25RetrieverConfig

logger = logging.getLogger(__name__)


class BM25Retriever(RetrieverInterface):
    """
    BM25Retriever 实现，支持独立的构建和加载过程。
    """

    INDEX_FILE_NAME = "bm25_retriever.pkl"

    def __init__(self, config: BM25RetrieverConfig, persist_dir: Path):
        self.config = config
        self.persist_dir = persist_dir
        # 【修复点 2】更新类型注解（可选，方便IDE提示）
        self._retriever: Optional[LangChainBM25Retriever] = None
        self.index_path = self.persist_dir / self.INDEX_FILE_NAME

    @property
    def is_initialized(self) -> bool:
        return self._retriever is not None

    def load_from_disk(self) -> bool:
        """尝试从磁盘加载索引。"""
        if not self.index_path.exists():
            logger.warning(
                f"BM25 index not found at '{self.index_path}'. It needs to be built via the ingestion process.")
            return False

        try:
            logger.info(f"Loading BM25Retriever index from '{self.index_path}'...")
            with open(self.index_path, "rb") as f:
                self._retriever = pickle.load(f)

            # 确保运行时使用的 top_k 是最新的配置
            self._retriever.k = self.config.top_k
            logger.info("BM25Retriever index loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25Retriever index. It might be corrupted. Please run ingestion.",
                         exc_info=e)
            return False

    def build_and_save(self, documents: List[Document]):
        """从文档构建索引并立即保存到磁盘。"""
        if not documents:
            logger.warning("No documents provided to build BM25 index.")
            return

        try:
            logger.info(f"Building new BM25 index from {len(documents)} documents...")

            # 【修复点 3】使用别名调用 LangChain 的静态方法
            self._retriever = LangChainBM25Retriever.from_documents(
                documents,
                k=self.config.top_k,
                bm25_params={"k1": self.config.k1, "b": self.config.b}
            )
            logger.info(f"BM25Retriever index built successfully. k1={self.config.k1}, b={self.config.b}")

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump(self._retriever, f)
            logger.info(f"BM25Retriever index saved to '{self.index_path}'")
        except Exception as e:
            raise DataProcessingError("Failed to build and save BM25 index", original_exception=e)

    def get_langchain_retriever(self) -> BaseRetriever:
        if not self.is_initialized:
            raise InitializationError("BM25Retriever",
                                      "Retriever is not initialized. Please load it from disk or build it first.")
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        if not self.is_initialized:
            raise InitializationError("BM25Retriever", "Retriever is not initialized.")
        try:
            return self._retriever.invoke(query)
        except Exception as e:
            raise RetrievalError(f"Error during BM25 retrieval for query: {query}", e)

    async def aretrieve(self, query: str) -> List[Document]:
        """【新增】异步检索"""
        if not self.is_initialized:
            raise InitializationError("BM25Retriever", "Retriever is not initialized.")
        try:
            # LangChain 自动将同步逻辑 wrap 到线程中
            return await self._retriever.ainvoke(query)
        except Exception as e:
            raise RetrievalError(f"Error during async BM25 retrieval for query: {query}", e)
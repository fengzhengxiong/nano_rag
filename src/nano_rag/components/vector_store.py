#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：vector_store.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:07 
'''

import logging
import asyncio
from pathlib import Path
from typing import List, Optional, Any
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..core.interfaces import VectorStoreInterface, EmbeddingInterface
from ..core.exceptions import InitializationError, DataProcessingError
from ..config.models import FaissVectorStoreConfig

logger = logging.getLogger(__name__)

class FaissVectorStore(VectorStoreInterface):
    """使用 FAISS 的向量数据库实现，支持 Async 和本地持久化。"""
    INDEX_FILE_NAME = "index.faiss"

    def __init__(self, config: FaissVectorStoreConfig, embedding_model: EmbeddingInterface, persist_dir: Path):
        self.config = config
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self._faiss_store: Optional[LangchainFAISS] = None
        logger.info(f"FAISS Vector Store configured. Persist directory: '{self.persist_dir}'")

    @property
    def is_initialized(self) -> bool:
        return self._faiss_store is not None

    def get_langchain_vectorstore(self) -> Optional[LangchainFAISS]:
        return self._faiss_store

    def load_local(self) -> bool:
        if self.is_initialized: return True
        index_path = self.persist_dir / self.INDEX_FILE_NAME
        if not index_path.exists(): return False
        try:
            self._faiss_store = LangchainFAISS.load_local(
                folder_path=str(self.persist_dir),
                embeddings=self.embedding_model.get_langchain_embeddings(),
                index_name="index",
                allow_dangerous_deserialization=self.config.allow_dangerous_deserialization
            )
            return True
        except Exception: return False

    def save_local(self) -> bool:
        if not self.is_initialized: return False
        try:
            self._faiss_store.save_local(folder_path=str(self.persist_dir), index_name="index")
            return True
        except Exception: return False

    def build_from_documents(self, documents: List[Document]):
        if not documents: return
        self._faiss_store = LangchainFAISS.from_documents(
            documents, self.embedding_model.get_langchain_embeddings()
        )
        self.save_local()

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """同步添加文档"""
        if not self.is_initialized:
            raise InitializationError("FaissVectorStore", "Store is not initialized.")
        if not documents: return []
        try:
            ids = self._faiss_store.add_documents(documents, **kwargs)
            self.save_local()
            return ids
        except Exception as e:
            raise DataProcessingError("Failed to add documents", e) from e

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """
        【新增】异步添加文档。
        使用 asyncio.to_thread 将 CPU 密集型操作移出主线程。
        """
        if not self.is_initialized:
             raise InitializationError("FaissVectorStore", "Store is not initialized.")
        if not documents: return []
        try:
            # 在线程池中执行添加和保存
            ids = await asyncio.to_thread(self._faiss_store.add_documents, documents, **kwargs)
            await asyncio.to_thread(self.save_local)
            return ids
        except Exception as e:
            raise DataProcessingError("Failed to async add documents", e) from e

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        if not self.is_initialized:
             raise InitializationError("FaissVectorStore", "Store not initialized.")
        return self._faiss_store.as_retriever(**kwargs)
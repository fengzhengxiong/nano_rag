#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：cache_service.py
@Author  ：fengzhengxiong
@Date    ：2025/12/30 10:08 
'''

import logging
import asyncio
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from ..core.interfaces import EmbeddingInterface

logger = logging.getLogger(__name__)


class SemanticCacheService:
    """
    基于向量相似度的语义缓存服务。
    原理：
    1. 将用户问题向量化。
    2. 在缓存专用的 FAISS 索引中搜索相似问题。
    3. 如果相似度 > 阈值 (如 0.9)，则直接返回之前存好的答案。
    """

    CACHE_INDEX_NAME = "semantic_cache"
    SCORE_THRESHOLD = 0.35  # FAISS L2 距离阈值 (越小越相似，0.35 约等于余弦相似度 0.9)

    def __init__(self, embedding_model: EmbeddingInterface, persist_dir: Path):
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir / "cache_store"
        self._store: Optional[FAISS] = None

        self._init_cache()

    def _init_cache(self):
        """初始化或加载缓存索引"""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        index_file = self.persist_dir / "index.faiss"

        try:
            if index_file.exists():
                logger.info("Loading semantic cache from disk...")
                self._store = FAISS.load_local(
                    folder_path=str(self.persist_dir),
                    embeddings=self.embedding_model.get_langchain_embeddings(),
                    index_name=self.CACHE_INDEX_NAME,
                    allow_dangerous_deserialization=True
                )
            else:
                logger.info("Creating new semantic cache index...")
                # 创建一个空的索引需要技巧：先创建一个 Dummy Document，或者延迟创建
                # 这里我们采用“懒加载”策略，第一次写入时再创建
                self._store = None
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Starting fresh.")
            self._store = None

    async def lookup(self, query: str) -> Optional[str]:
        """
        (Async) 查找缓存。
        返回: 缓存的答案字符串，如果未命中则返回 None。
        """
        if self._store is None:
            return None

        try:
            # 在线程池中搜索，避免阻塞
            # search_type="similarity_score_threshold" 在 FAISS 里实现比较绕
            # 我们直接用 similarity_search_with_score
            docs_and_scores = await asyncio.to_thread(
                self._store.similarity_search_with_score,
                query,
                k=1
            )

            if not docs_and_scores:
                return None

            doc, score = docs_and_scores[0]

            # FAISS L2 距离：越小越好。0 是完全一样。
            # 经验值：0.3 以下通常非常相似。
            logger.debug(f"Cache hit check: score={score:.4f} (Threshold: {self.SCORE_THRESHOLD})")

            if score < self.SCORE_THRESHOLD:
                logger.info(f"⚡️ Semantic Cache HIT! (Score: {score:.4f})")
                return doc.metadata.get("answer")

            return None

        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
            return None

    async def update(self, query: str, answer: str):
        """
        (Async) 更新缓存。
        """
        try:
            doc = Document(
                page_content=query,  # 索引的是“问题”
                metadata={"answer": answer}  # 存储的是“答案”
            )

            if self._store is None:
                # 第一次创建
                self._store = await asyncio.to_thread(
                    FAISS.from_documents,
                    [doc],
                    self.embedding_model.get_langchain_embeddings()
                )
            else:
                await asyncio.to_thread(self._store.add_documents, [doc])

            # 立即持久化 (生产环境可以改为定时异步持久化)
            await asyncio.to_thread(
                self._store.save_local,
                str(self.persist_dir),
                self.CACHE_INDEX_NAME
            )
            logger.debug("Cache updated and saved.")

        except Exception as e:
            logger.error(f"Cache update failed: {e}")
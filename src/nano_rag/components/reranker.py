#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：reranker.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:07 
'''

import os
import logging
import torch
import asyncio
from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from ..core.interfaces import RerankerInterface
from ..core.exceptions import InitializationError, RetrievalError
from ..config.models import BGERerankerConfig

logger = logging.getLogger(__name__)


class BGEReranker(RerankerInterface):
    """
    基于 BGE-Reranker (Cross-Encoder) 的重排序实现。
    """

    def __init__(self, config: BGERerankerConfig):
        # 【修复点 1】存入私有变量 _config，避免与 @property 冲突
        self._config = config
        self._model = None

        try:
            logger.info(f"Initializing BGE Reranker model: {config.model_name}...")

            if os.path.sep in config.model_name or os.path.exists(config.model_name):
                if not os.path.exists(config.model_name):
                    raise InitializationError("BGEReranker", f"Local model directory not found: {config.model_name}")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 【修复点 2】消除 automodel_args 弃用警告
            # sentence-transformers v3 推荐使用 model_name_or_path
            self._model = CrossEncoder(
                model_name=config.model_name,  # v3内部会自动映射，或者你可以显式改为 model_name_or_path=config.model_name
                max_length=512,
                device=device,
                # 显式传递 local_files_only 给 transformers 后端
                model_kwargs={"local_files_only": True}
            )
            logger.info(f"BGE Reranker initialized on device: {device}")
        except Exception as e:
            # 降级重试逻辑
            try:
                logger.warning(f"First attempt failed ({e}), trying without explicit local_files_only arg...")
                self._model = CrossEncoder(
                    model_name=config.model_name,
                    max_length=512,
                    device=device
                )
                logger.info(f"BGE Reranker initialized (fallback mode) on device: {device}")
            except Exception as e2:
                raise InitializationError("BGEReranker", f"Failed to load model {config.model_name}", e2)

    # 【修复点 3】正确实现接口定义的 property
    @property
    def config(self) -> BGERerankerConfig:
        return self._config

    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """同步重排序"""
        if not documents:
            return []
        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self._model.predict(pairs)

            results = list(zip(documents, scores))
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            raise RetrievalError("Reranking process failed", e)

    async def arerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """异步重排序"""
        if not documents:
            return []
        try:
            return await asyncio.to_thread(self.rerank, query, documents)
        except Exception as e:
            raise RetrievalError("Async reranking process failed", e)
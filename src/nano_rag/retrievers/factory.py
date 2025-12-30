#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：factory.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:04 
'''

import logging
from pathlib import Path

# 【重构】导入路径已全部更新
from ..core.interfaces import RetrieverInterface, VectorStoreInterface
from ..config.models import RetrievalStrategyConfig
from ..core.exceptions import ConfigurationError, InitializationError

from .bm25 import BM25Retriever
from .vector import VectorRetriever
from .hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """
    根据配置创建和组装各种 Retriever 的工厂类。
    """

    @staticmethod
    def create(
            config: RetrievalStrategyConfig,
            vector_store: VectorStoreInterface,
            persist_dir: Path
    ) -> RetrieverInterface:
        """
        工厂主方法，根据配置的 'strategy' 字段创建检索器。
        """
        strategy = config.retriever.strategy
        logger.info(f"Creating retriever with strategy: '{strategy}'")

        if strategy == "vector":
            return VectorRetriever(config.retriever.vector_config, vector_store)

        elif strategy == "bm25":
            bm25_retriever = BM25Retriever(config.retriever.bm25_config, persist_dir)
            if not bm25_retriever.load_from_disk():
                # 在查询流程中，如果BM25索引不存在，我们不能继续，只能抛出异常
                raise InitializationError("RetrieverFactory",
                                          "BM25 index not found. Please run the data ingestion process first.")
            return bm25_retriever

        elif strategy == "hybrid":
            hybrid_config = config.retriever.hybrid_config
            sub_retrievers = []

            # 1. 创建并加载 BM25 检索器
            bm25_retriever = BM25Retriever(config.retriever.bm25_config, persist_dir)
            if not bm25_retriever.load_from_disk():
                raise InitializationError("RetrieverFactory",
                                          "Hybrid strategy requires a pre-built BM25 index. Please run ingestion.")
            sub_retrievers.append(bm25_retriever)

            # 2. 创建 Vector 检索器
            vector_retriever = VectorRetriever(config.retriever.vector_config, vector_store)
            sub_retrievers.append(vector_retriever)

            # 3. 创建混合检索器
            weights = [hybrid_config.bm25_weight, hybrid_config.vector_weight]
            return HybridRetriever(hybrid_config, sub_retrievers, weights)

        else:
            raise ConfigurationError(f"Unsupported retriever strategy: '{strategy}'")
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：application.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:00 
'''

import logging

from .config.models import ResolvedConfig
from .core.exceptions import InitializationError
from .factories import ComponentFactory
from .retrievers.factory import RetrieverFactory
from .services.ingestion_service import IngestionService
from .services.query_service import QueryService

logger = logging.getLogger(__name__)


class RAGApplication:
    """
    应用容器，负责在启动时通过工厂初始化和组装所有组件和服务。
    这是整个应用程序的单一入口点和状态持有者。
    """

    def __init__(self, config: ResolvedConfig):
        """
        使用已解析的配置对象初始化应用程序。
        """
        self.config = config
        self.ingestion_service: IngestionService | None = None
        self.query_service: QueryService | None = None

        logger.info("Initializing RAGApplication...")
        try:
            self._initialize_and_assemble()
            logger.info("RAGApplication initialized successfully.")
        except Exception as e:
            # 捕获在组装过程中的任何异常，并将其包装为 InitializationError
            logger.critical(f"Application failed to initialize: {e}", exc_info=True)
            raise InitializationError("RAGApplication", "Fatal error during application startup.", e) from e

    def _initialize_and_assemble(self):
        """
        [架构核心]
        使用工厂模式创建所有组件，然后将它们组装成服务。
        """
        logger.info("Assembling application components and services...")

        # --- 1. 创建无依赖或只有配置依赖的基础组件 ---
        document_loader = ComponentFactory.create_document_loader(
            config=self.config.data_source,
            data_dir=self.config.paths.data_dir
        )

        text_splitter = ComponentFactory.create_text_splitter(
            config=self.config.text_splitter
        )

        embedding_model = ComponentFactory.create_embedding_model(
            config=self.config.embedding
        )

        llm = ComponentFactory.create_llm(
            config=self.config.llm
        )

        # --- 2. 创建依赖其他组件的组件 ---
        vector_store = ComponentFactory.create_vector_store(
            config=self.config.vector_store,
            embedding_model=embedding_model,
            persist_dir=self.config.resolved_paths.persist_dir
        )

        # --- 3. 创建最顶层的复杂组件 (Retriever & Reranker) ---

        # A. 创建 Retriever (原有代码)
        retriever = None
        try:
            retriever = RetrieverFactory.create(
                config=self.config.retrieval_strategy,
                vector_store=vector_store,
                persist_dir=self.config.resolved_paths.persist_dir
            )
        except InitializationError as e:
            logger.warning(f"Retriever initialization skipped: {e}")

        # B. 【新增】创建 Reranker
        reranker = None
        # 检查配置中是否启用了 reranker (check if self.config.retrieval_strategy.reranker is not None)
        reranker_profile_config = self.config.retrieval_strategy.reranker

        if reranker_profile_config:
            # 让我们简化一下，直接在 application 里根据 active profile 创建
            active_name = reranker_profile_config.active
            if active_name in reranker_profile_config.profiles:
                reranker_config = reranker_profile_config.profiles[active_name]
                try:
                    reranker = ComponentFactory.create_reranker(reranker_config)
                except Exception as e:
                    logger.warning(f"Failed to initialize Reranker: {e}. Continuing without reranker.")
            else:
                logger.warning(f"Active reranker profile '{active_name}' not found in profiles.")

        # --- 4. 组装服务 ---

        # IngestionService 不需要 retriever 实例，只需要配置，所以总是可以创建
        self.ingestion_service = IngestionService(
            retrieval_config=self.config.retrieval_strategy,
            document_loader=document_loader,
            text_splitter=text_splitter,
            vector_store=vector_store,
            persist_dir=self.config.resolved_paths.persist_dir
        )

        # QueryService 只有在 retriever 成功创建时才创建
        if retriever:
            self.query_service = QueryService(
                llm=llm,
                retriever=retriever,
                reranker=reranker
            )
        else:
            self.query_service = None
            logger.info("QueryService is not initialized (waiting for data ingestion).")
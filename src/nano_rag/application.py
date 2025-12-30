#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：application.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:00 
'''

import logging
import os

from .config.models import ResolvedConfig
from .core.exceptions import InitializationError
from .factories import ComponentFactory
from .retrievers.factory import RetrieverFactory
from .services.ingestion_service import IngestionService
from .services.query_service import QueryService
from .services.cache_service import SemanticCacheService

from .config.loader import load_app_config
from .config.prompt_loader import load_prompts

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
        self.cache_service : SemanticCacheService | None = None

        # 【核心修改】在初始化组件之前，先设置可观测性
        self._setup_observability()

        logger.info("Initializing RAGApplication...")
        try:
            self._initialize_and_assemble()
            logger.info("RAGApplication initialized successfully.")
        except Exception as e:
            # 捕获在组装过程中的任何异常，并将其包装为 InitializationError
            logger.critical(f"Application failed to initialize: {e}", exc_info=True)
            raise InitializationError("RAGApplication", "Fatal error during application startup.", e) from e

    def _setup_observability(self):
        """
        根据配置自动开启 LangSmith 追踪。
        注意：这里我们需要重新加载一下原始 AppConfig 来获取 observability 字段，
        或者你也可以修改 resolve_active_configs 把它透传给 ResolvedConfig。
        为了简单，我们直接在这里读一次原始配置的对应部分。
        """
        try:
            # 这里的逻辑稍微有点 tricky，因为 config 已经是 ResolvedConfig 了
            # 我们假设你在 ResolvedConfig 里没加 observability
            # 所以我们可以重新读一下，或者更简单的：
            # 建议你在上一步把 observability 也加到 ResolvedConfig 里
            # 如果没加，我们可以通过 load_app_config() 拿

            raw_config = load_app_config()  # 这会读取 default_config.yaml
            obs_config = raw_config.observability

            if obs_config and obs_config.enabled:
                logger.info(f"🔭 Enabling LangSmith Tracing (Project: {obs_config.project_name})")

                # 设置 LangChain 官方要求的环境变量
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
                os.environ["LANGCHAIN_PROJECT"] = obs_config.project_name

                if obs_config.api_key:
                    os.environ["LANGCHAIN_API_KEY"] = obs_config.api_key
            else:
                logger.info("🔭 Observability is disabled.")

        except Exception as e:
            logger.warning(f"Failed to setup observability: {e}")

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

        self.cache_service = SemanticCacheService(
            embedding_model=embedding_model,
            persist_dir=self.config.resolved_paths.persist_dir
        )

        # 【新增】加载 Prompt 配置
        prompt_config = load_prompts()
        logger.info("Loaded external prompt configuration.")

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
                reranker=reranker,
                cache_service=self.cache_service,
                prompt_config=prompt_config  # 【新增】注入
            )
        else:
            self.query_service = None
            logger.info("QueryService is not initialized (waiting for data ingestion).")
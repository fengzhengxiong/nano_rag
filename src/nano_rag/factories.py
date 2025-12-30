#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：factories.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:00 
'''

import logging
from pathlib import Path
from typing import Union

from .core.exceptions import ConfigurationError
from .components.reranker import BGEReranker
from .components.reranker_onnx import ONNXBGEReranker
from .components.embedding import OllamaEmbedding, HuggingFaceEmbedding
from .components.document_loader import DirectoryLoader
from .components.text_splitter import RecursiveCharacterTextSplitter
from .components.vector_store import FaissVectorStore
from .components.llm import OllamaLLM, OpenAILLM
from .core.interfaces import (
    DocumentLoaderInterface, TextSplitterInterface, EmbeddingInterface,
    VectorStoreInterface, LLMInterface, RerankerInterface
)
from .config.models import (
    DirectoryLoaderConfig, RecursiveCharacterTextSplitterConfig,
    OllamaEmbeddingConfig, FaissVectorStoreConfig, OllamaLLMConfig,  BGERerankerConfig,
    HuggingFaceEmbeddingConfig, OpenAILLMConfig
)

logger = logging.getLogger(__name__)

class ComponentFactory:
    """
    一个统一的工厂类，用于根据配置创建所有基础组件实例。
    """

    @staticmethod
    def create_document_loader(config: DirectoryLoaderConfig, data_dir: Path) -> DocumentLoaderInterface:
        """创建文档加载器。"""
        logger.info(f"Creating DocumentLoader of type: {config.type}")
        if config.type == "directory_loader":
            return DirectoryLoader(config, data_dir)
        raise ConfigurationError(f"Unsupported document loader type: {config.type}")

    @staticmethod
    def create_text_splitter(config: RecursiveCharacterTextSplitterConfig) -> TextSplitterInterface:
        """创建文本分割器。"""
        logger.info(f"Creating TextSplitter of type: {config.type}")
        if config.type == "recursive_character":
            return RecursiveCharacterTextSplitter(config)
        raise ConfigurationError(f"Unsupported text splitter type: {config.type}")

    @staticmethod
    def create_embedding_model(config: Union[OllamaEmbeddingConfig, HuggingFaceEmbeddingConfig]) -> EmbeddingInterface:
        """创建向量嵌入模型。"""
        logger.info(f"Creating Embedding model of type: {config.type}")

        if config.type == "ollama_embedding":
            return OllamaEmbedding(config)
        # 【新增】支持本地 HF 模型
        elif config.type == "huggingface":
            return HuggingFaceEmbedding(config)

        raise ConfigurationError(f"Unsupported embedding model type: {config.type}")

    @staticmethod
    def create_vector_store(
        config: FaissVectorStoreConfig,
        embedding_model: EmbeddingInterface,
        persist_dir: Path
    ) -> VectorStoreInterface:
        """创建向量数据库。"""
        logger.info(f"Creating VectorStore of type: {config.type}")
        if config.type == "faiss":
            # 向量存储在创建时，就应该尝试从本地加载
            store = FaissVectorStore(config, embedding_model, persist_dir)
            store.load_local()
            return store
        raise ConfigurationError(f"Unsupported vector store type: {config.type}")

    @staticmethod
    def create_llm(config: Union[OllamaLLMConfig, OpenAILLMConfig]) -> LLMInterface:
        """创建大语言模型。"""
        logger.info(f"Creating LLM of type: {config.type}")

        if config.type == "ollama":
            return OllamaLLM(config)
        # 【新增】支持 OpenAI/硅基流动
        elif config.type == "openai":
            return OpenAILLM(config)

        raise ConfigurationError(f"Unsupported LLM type: {config.type}")

    # @staticmethod
    # def create_reranker(config: BGERerankerConfig) -> RerankerInterface:
    #     """创建重排序器。"""
    #     logger.info(f"Creating Reranker of type: {config.type}")
    #     if config.type == "bge_reranker":
    #         return BGEReranker(config)
    #     raise ConfigurationError(f"Unsupported reranker type: {config.type}")

    @staticmethod
    def create_reranker(config: BGERerankerConfig) -> RerankerInterface:
        """创建重排序器 (支持 PyTorch 和 ONNX 双后端)。"""
        logger.info(f"Creating Reranker of type: {config.type}, Backend: {config.backend}")

        if config.type == "bge_reranker":
            # 【核心逻辑】根据后端配置选择实现类
            if config.backend == "onnx":
                return ONNXBGEReranker(config)
            else:
                return BGEReranker(config)

        raise ConfigurationError(f"Unsupported reranker type: {config.type}")
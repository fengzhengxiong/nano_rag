#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：embedding.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:06 
'''

import logging
from pathlib import Path
from typing import List, Any

# 【新增】引入项目根目录查找工具
from ..utils.helpers import find_project_root

from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from ..core.interfaces import EmbeddingInterface
from ..core.exceptions import InitializationError
from ..config.models import OllamaEmbeddingConfig, HuggingFaceEmbeddingConfig

logger = logging.getLogger(__name__)


class OllamaEmbedding(EmbeddingInterface):
    """使用 Ollama 的向量嵌入模型实现。"""

    def __init__(self, config: OllamaEmbeddingConfig):
        self.config = config
        try:
            init_params = config.model_dump(exclude=["type"])

            if "model_name" in init_params:
                init_params["model"] = init_params.pop("model_name")

            self._embeddings = OllamaEmbeddings(**init_params)
            logger.info(f"Ollama Embeddings initialized with model: '{config.model_name}'.")
        except Exception as e:
            raise InitializationError("OllamaEmbedding",
                                      f"Failed to initialize Ollama embedding model '{config.model_name}'", e) from e

    def get_langchain_embeddings(self) -> OllamaEmbeddings:
        return self._embeddings

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)


class HuggingFaceEmbedding(EmbeddingInterface):
    """使用本地 HuggingFace 模型的 Embedding 实现。"""

    def __init__(self, config: HuggingFaceEmbeddingConfig):
        self.config = config
        try:
            logger.info(f"Initializing Local Embedding from config: {config.model_name}")

            # 【关键修复】将相对路径转换为绝对路径
            model_path = Path(config.model_name)
            if not model_path.is_absolute():
                # 如果是相对路径，假设是相对于项目根目录
                project_root = find_project_root()
                model_path = project_root / model_path

            # 检查路径是否存在
            if not model_path.exists():
                raise FileNotFoundError(f"Local model directory not found at: {model_path}")

            logger.info(f"Resolved absolute model path: {model_path}")

            self._embeddings = HuggingFaceEmbeddings(
                # 传入处理后的绝对路径字符串
                model_name=str(model_path),
                model_kwargs={'device': config.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Local HuggingFace Embedding initialized. Device: {config.device}")
        except Exception as e:
            # 打印更详细的错误信息
            raise InitializationError("HuggingFaceEmbedding", f"Failed to load model from '{config.model_name}'", e)

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        return self._embeddings

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        return self._embeddings.embed_query(text)
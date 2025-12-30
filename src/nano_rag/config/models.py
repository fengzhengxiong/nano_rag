#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：models.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:06 
'''

from pydantic import BaseModel, Field, DirectoryPath, ConfigDict
from typing import Literal, Optional, List, Dict, Generic, TypeVar, Union
from pathlib import Path

# ==============================================================================
# PART 1: 原始配置文件模型 (Raw Config Models)
# ==============================================================================
ConfigT = TypeVar('ConfigT')

class ComponentModel(BaseModel, Generic[ConfigT]):
    """定义一个组件，包含类型和具体配置"""
    type: str
    config: ConfigT

class ProfileSet(BaseModel, Generic[ConfigT]):
    """定义一组 profiles，其中一个为 active"""
    active: str
    profiles: Dict[str, ConfigT]

# ==============================================================================
# PART 2: 各组件具体配置的 Pydantic 模型
# ==============================================================================
class LoggingConfig(BaseModel):
    log_dir: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    max_bytes: int
    backup_count: int

class PathsConfig(BaseModel):
    data_dir: DirectoryPath
    cache_db_file: Path
    persist_base_dir: Path
    # persist_base_dir: DirectoryPath

# --- LLM Configs ---
class OllamaLLMConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    type: Literal["ollama"]
    base_url: str
    model_name: str
    temperature: float = 0.1
    num_ctx: int

# 【新增】OpenAI/硅基流动 配置
class OpenAILLMConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    type: Literal["openai"]
    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 2048

# --- Embedding Configs ---
class OllamaEmbeddingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    type: Literal["ollama_embedding"]
    base_url: str
    model_name: str

# 【新增】本地 HuggingFace Embedding 配置
class HuggingFaceEmbeddingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    type: Literal["huggingface"]
    model_name: str  # 本地路径
    device: str = "cpu" # cpu 或 cuda

class FaissVectorStoreConfig(BaseModel):
    type: Literal["faiss"]
    allow_dangerous_deserialization: bool

class DirectoryLoaderConfig(BaseModel):
    type: Literal["directory_loader"]
    glob_pattern: str
    use_multreading: bool
    silent_errors: bool
    on_unsupported_type: Literal["ignore", "warn", "error"]
    loader_mapping: Dict[str, str]
    text_loader_autodetect_encoding: bool

class RecursiveCharacterTextSplitterConfig(BaseModel):
    type: Literal["recursive_character"]
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    keep_separator: bool
    separators: List[str]

class BGERerankerConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    type: Literal["bge_reranker"]
    model_name: str
    use_fp16: bool
    top_k: int = Field(gt=0)

class BM25RetrieverConfig(BaseModel):
    k1: float
    b: float
    top_k: int = Field(gt=0)

class VectorRetrieverConfig(BaseModel):
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"]
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: int = Field(gt=0)

class HybridRetrieverConfig(BaseModel):
    bm25_weight: float = Field(ge=0.0, le=1.0)
    vector_weight: float = Field(ge=0.0, le=1.0)
    top_k: int = Field(gt=0)

class RetrieverConfig(BaseModel):
    strategy: Literal["vector", "bm25", "hybrid"]
    bm25_config: Optional[BM25RetrieverConfig] = None
    vector_config: Optional[VectorRetrieverConfig] = None
    hybrid_config: Optional[HybridRetrieverConfig] = None

class RerankerProfile(BaseModel):
    active: str
    profiles: Dict[str, BGERerankerConfig] # 目前只支持BGE，未来可扩展

class RetrievalStrategyConfig(BaseModel):
    retriever: RetrieverConfig
    reranker: Optional[RerankerProfile] = None

class CacheConfig(BaseModel):
    enable: bool
    type: Optional[Literal["sqlite", "memory"]] = None

# ==============================================================================
# PART 3: 完整的原始应用配置模型 (AppConfig)
# ==============================================================================
class AppConfig(BaseModel):
    """映射整个 YAML 文件的结构"""
    logging: LoggingConfig
    paths: PathsConfig
    data_source: ProfileSet[DirectoryLoaderConfig]
    text_splitter: ProfileSet[RecursiveCharacterTextSplitterConfig]

    # 【修改】使用 Union 支持多种 Embedding 配置
    embedding: ProfileSet[Union[OllamaEmbeddingConfig, HuggingFaceEmbeddingConfig]]

    vector_store: ProfileSet[FaissVectorStoreConfig]

    # 【修改】使用 Union 支持多种 LLM 配置
    llm: ProfileSet[Union[OllamaLLMConfig, OpenAILLMConfig]]

    cache: ProfileSet[CacheConfig]
    retrieval_strategy: ProfileSet[RetrievalStrategyConfig]

# ==============================================================================
# PART 4: 最终解析后的扁平化配置模型 (ResolvedConfig)
# ==============================================================================
class ResolvedPaths(BaseModel):
    persist_dir: DirectoryPath

class ResolvedConfig(BaseModel):
    """AppConfig 被解析后，供应用程序直接使用的扁平化配置"""
    model_config = ConfigDict(protected_namespaces=())  # 顺手加上这个，防止以后报警告

    logging: LoggingConfig
    paths: PathsConfig
    data_source: DirectoryLoaderConfig
    text_splitter: RecursiveCharacterTextSplitterConfig

    # 【修改点 1】改为 Union，支持多种 Embedding 配置
    embedding: Union[OllamaEmbeddingConfig, HuggingFaceEmbeddingConfig]

    vector_store: FaissVectorStoreConfig

    # 【修改点 2】改为 Union，支持多种 LLM 配置
    llm: Union[OllamaLLMConfig, OpenAILLMConfig]

    cache: CacheConfig
    retrieval_strategy: RetrievalStrategyConfig
    resolved_paths: ResolvedPaths
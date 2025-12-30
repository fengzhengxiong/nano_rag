#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：interfaces.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:05 
'''

from abc import ABC, abstractmethod
from typing import List, Iterator, Any, Optional, AsyncIterator, Tuple

# 引入 Chat 消息定义
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever as LangchainBaseRetriever
from langchain_core.vectorstores import VectorStore as LangchainVectorStore


# ==============================================================================
# Document Loader Interface
# ==============================================================================
class DocumentLoaderInterface(ABC):
    @abstractmethod
    def load(self) -> List[Document]: pass

    @abstractmethod
    def lazy_load(self) -> Iterator[Document]: pass

    @abstractmethod
    def load_single_file(self, file_path: Any) -> List[Document]: pass


# ==============================================================================
# Text Splitter Interface
# ==============================================================================
class TextSplitterInterface(ABC):
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]: pass


# ==============================================================================
# Embedding Interface
# ==============================================================================
class EmbeddingInterface(ABC):
    @abstractmethod
    def get_langchain_embeddings(self) -> LangchainEmbeddings: pass

    @abstractmethod
    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]: pass

    @abstractmethod
    def embed_query(self, text: str, **kwargs: Any) -> List[float]: pass

    @property
    @abstractmethod
    def model_name(self) -> str: pass


# ==============================================================================
# Vector Store Interface
# ==============================================================================
class VectorStoreInterface(ABC):
    @abstractmethod
    def get_langchain_vectorstore(self) -> Optional[LangchainVectorStore]: pass

    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]: pass

    @abstractmethod
    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """(Async) 异步添加文档"""
        pass

    @abstractmethod
    def build_from_documents(self, documents: List[Document]) -> None: pass

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> LangchainBaseRetriever: pass

    @abstractmethod
    def load_local(self) -> bool: pass

    @abstractmethod
    def save_local(self) -> bool: pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool: pass


# ==============================================================================
# Retriever Interface
# ==============================================================================
class RetrieverInterface(ABC):
    @abstractmethod
    def get_langchain_retriever(self) -> LangchainBaseRetriever: pass

    @abstractmethod
    def retrieve(self, query: str) -> List[Document]: pass

    @abstractmethod
    async def aretrieve(self, query: str) -> List[Document]:
        """(Async) 异步检索"""
        pass


# ==============================================================================
# Reranker Interface
# ==============================================================================
class RerankerInterface(ABC):
    @property
    @abstractmethod
    def config(self) -> Any: pass

    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]: pass

    @abstractmethod
    async def arerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """(Async) 异步重排序"""
        pass


# ==============================================================================
# LLM Interface (必须更新！)
# ==============================================================================
class LLMInterface(ABC):
    """
    大语言模型接口。
    注意：已移除 generate 方法，改用 invoke/ainvoke (Chat模式)。
    """

    @abstractmethod
    def get_langchain_llm(self) -> BaseChatModel:
        """返回底层的 Langchain ChatModel 实例"""
        pass

    @abstractmethod
    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> str:
        """同步调用：输入消息列表，返回字符串内容"""
        pass

    @abstractmethod
    async def ainvoke(self, messages: List[BaseMessage], **kwargs: Any) -> str:
        """(Async) 异步调用：输入消息列表，返回字符串内容"""
        pass

    @abstractmethod
    def stream(self, messages: List[BaseMessage], **kwargs: Any) -> Iterator[str]:
        """同步流式"""
        pass

    @abstractmethod
    async def astream(self, messages: List[BaseMessage], **kwargs: Any) -> AsyncIterator[str]:
        """(Async) 异步流式"""
        pass
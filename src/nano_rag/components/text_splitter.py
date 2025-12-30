#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：text_splitter.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:07 
'''

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter as LCRecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..core.interfaces import TextSplitterInterface
from ..core.exceptions import InitializationError, DataProcessingError
from ..config.models import RecursiveCharacterTextSplitterConfig

logger = logging.getLogger(__name__)

class RecursiveCharacterTextSplitter(TextSplitterInterface):
    """使用 Langchain 的 RecursiveCharacterTextSplitter 实现文本分割。"""

    def __init__(self, config: RecursiveCharacterTextSplitterConfig):
        self.config = config
        try:
            init_params = config.model_dump(exclude=["type"])

            # 【修复点 2】使用别名来实例化 LangChain 的对象
            self._splitter = LCRecursiveCharacterTextSplitter(**init_params)

            logger.info(
                f"RecursiveCharacterTextSplitter initialized. Chunk size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
        except Exception as e:
            raise InitializationError("RecursiveCharacterTextSplitter", "Failed to create splitter instance", e) from e

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        try:
            return self._splitter.split_documents(documents)
        except Exception as e:
            raise DataProcessingError("Failed to split documents", original_exception=e) from e
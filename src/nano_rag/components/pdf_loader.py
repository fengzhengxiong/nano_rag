#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：pdf_loader.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 15:57 
'''

import logging
from typing import List, Iterator
from pathlib import Path

from langchain_core.documents import Document
from docling.document_converter import DocumentConverter

from ..core.interfaces import DocumentLoaderInterface
from ..core.exceptions import DataProcessingError

logger = logging.getLogger(__name__)


class AdvancedPDFLoader(DocumentLoaderInterface):
    """
    【核心组件】高级 PDF 加载器 (基于 IBM Docling)。

    功能：
    1. 使用视觉模型识别文档布局（标题、段落、表格）。
    2. 将复杂的 PDF 表格完美转换为 Markdown Table 格式。
    3. 输出结构化的 Markdown 文本，极大提升 LLM 对数据的理解力。
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        # 初始化转换器
        # 注意：首次运行时，Docling 会自动下载 OCR 模型到本地缓存
        try:
            self._converter = DocumentConverter()
        except Exception as e:
            raise DataProcessingError("Failed to initialize Docling Converter", e)

    def load(self) -> List[Document]:
        """一次性加载"""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """
        核心逻辑：PDF -> Docling -> Markdown -> LangChain Document
        """
        try:
            logger.info(f"🚀 [Docling] Starting deep parsing for: {self.file_path.name} ...")

            # 1. 执行转换 (耗时操作，取决于文件大小和机器性能)
            conversion_result = self._converter.convert(str(self.file_path))

            # 2. 导出为 Markdown
            # 这是魔法所在：Docling 会把表格变成 | Header | Value | 这种格式
            md_content = conversion_result.document.export_to_markdown()

            if not md_content.strip():
                logger.warning(f"[Docling] Parsed content is empty for {self.file_path.name}")
                return

            logger.info(f"✅ [Docling] Successfully parsed {self.file_path.name}. Content length: {len(md_content)}")

            # 3. 封装为 Document 对象
            # 我们在 metadata 里标记来源和解析器类型
            yield Document(
                page_content=md_content,
                metadata={
                    "source": str(self.file_path),
                    "filename": self.file_path.name,
                    "parser": "docling_v2_markdown"
                }
            )

        except Exception as e:
            logger.error(f"❌ [Docling] Failed to parse {self.file_path}: {e}")
            # 这里我们选择抛出异常，因为如果 PDF 解析失败，通常意味着数据源有问题
            raise DataProcessingError(f"Docling parsing failed for {self.file_path}", e)

    def load_single_file(self, file_path: Path) -> List[Document]:
        """接口适配：加载单个文件"""
        self.file_path = file_path
        return self.load()
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：document_loader.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:06 
'''

import logging
from typing import List, Iterator, Type, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.document_loaders.base import BaseLoader

from ..core.interfaces import DocumentLoaderInterface
from ..core.exceptions import ConfigurationError, DataProcessingError
from ..config.models import DirectoryLoaderConfig
from ..utils.helpers import dynamic_import

# 【新增】导入刚才写好的高级加载器
from .pdf_loader import AdvancedPDFLoader

logger = logging.getLogger(__name__)


class DirectoryLoader(DocumentLoaderInterface):
    """
    智能目录加载器。
    策略：
    1. PDF 文件 -> 使用 AdvancedPDFLoader (Docling) 进行深度解析。
    2. 其他文件 -> 根据配置动态加载 (如 .txt, .md 使用 TextLoader)。
    """

    def __init__(self, config: DirectoryLoaderConfig, data_dir: Path):
        self.config = config
        self.data_dir = data_dir

        if not self.data_dir.exists() or not self.data_dir.is_dir():
            raise ConfigurationError(f"Data directory '{self.data_dir}' does not exist.")
        logger.info(f"DirectoryLoader configured. Root: '{self.data_dir}'")

    def _get_loader_instance_for_file(self, file_path: Path) -> Any:
        """工厂方法：根据文件后缀返回具体的 Loader 实例"""
        file_ext = file_path.suffix.lower()

        # --- 策略路由 ---

        # 1. 优先处理 PDF (使用 Docling)
        if file_ext == ".pdf":
            logger.debug(f"Routing {file_path.name} to AdvancedPDFLoader (Docling)")
            return AdvancedPDFLoader(file_path)

        # 2. 其他类型查表 (Config driven)
        class_path = self.config.loader_mapping.get(file_ext)

        if not class_path:
            if self.config.on_unsupported_type == "warn":
                logger.warning(f"No loader configured for '{file_ext}'. Skipping {file_path}")
            elif self.config.on_unsupported_type == "error":
                raise DataProcessingError(f"Unsupported file type '{file_ext}' for file: {file_path}")
            return None

        # 3. 动态实例化标准 LangChain Loader
        try:
            loader_cls: Type[BaseLoader] = dynamic_import(class_path)
            loader_kwargs = {}

            # 自动编码检测处理
            is_text_or_csv = any(name in loader_cls.__name__ for name in ["TextLoader", "CSVLoader"])
            if is_text_or_csv and self.config.text_loader_autodetect_encoding:
                try:
                    loader_kwargs["autodetect_encoding"] = True
                except TypeError:
                    pass

            return loader_cls(str(file_path), **loader_kwargs)

        except Exception as e:
            logger.error(f"Failed to instantiate loader for {file_path}: {e}")
            return None

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        logger.info(f"Starting ingestion from '{self.data_dir}'...")
        file_paths = list(self.data_dir.rglob(self.config.glob_pattern))

        for file_path in file_paths:
            if not file_path.is_file(): continue

            try:
                loader = self._get_loader_instance_for_file(file_path)
                if loader:
                    # 如果是 LangChain 标准 Loader，调用 lazy_load
                    if hasattr(loader, "lazy_load"):
                        yield from loader.lazy_load()
                    # 如果是我们自定义的 Docling Loader (虽然它也实现了 lazy_load，但为了保险)
                    elif hasattr(loader, "load"):
                        yield from loader.load()
            except Exception as e:
                if self.config.silent_errors:
                    logger.error(f"Error loading file {file_path}, skipping: {e}")
                else:
                    raise DataProcessingError(f"Error loading {file_path}", e) from e

    def load_single_file(self, file_path: Path) -> List[Document]:
        """增量更新专用接口"""
        if not file_path.is_file(): return []
        try:
            loader = self._get_loader_instance_for_file(file_path)
            if loader:
                if hasattr(loader, "load"):
                    return loader.load()
                elif hasattr(loader, "lazy_load"):
                    return list(loader.lazy_load())
            return []
        except Exception as e:
            logger.error(f"Error loading single file {file_path}: {e}")
            return []
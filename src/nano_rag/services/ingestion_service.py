#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：ingestion_service.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:02 
'''

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

# 【重构】导入路径已全部更新
from ..core.interfaces import DocumentLoaderInterface, TextSplitterInterface, VectorStoreInterface
from ..core.exceptions import DataProcessingError
from ..config.models import RetrievalStrategyConfig
from ..retrievers.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


class IngestionService:
    """
    负责数据注入流程：检测变更、加载、分割、更新向量库并构建BM25索引。
    """

    def __init__(
            self,
            retrieval_config: RetrievalStrategyConfig,
            document_loader: DocumentLoaderInterface,
            text_splitter: TextSplitterInterface,
            vector_store: VectorStoreInterface,
            persist_dir: Path
    ):
        self.retrieval_config = retrieval_config
        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self.vector_store = vector_store
        self.persist_dir = persist_dir

        # 元数据文件用于追踪已处理文件的状态
        self.metadata_path = self.persist_dir / "ingestion_metadata.json"

    def _get_file_hash(self, file_path: Path) -> str:
        """计算文件的SHA256哈希值。"""
        h = hashlib.sha256()
        with file_path.open("rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def _load_metadata(self) -> Dict[str, str]:
        """加载已处理文件的元数据（文件路径 -> 哈希值）。"""
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: Dict[str, str]):
        """保存元数据。"""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def run(self, force_rebuild: bool = False):
        """
        执行完整的数据注入流程。
        """
        logger.info("Starting data ingestion process...")
        processed_files_meta = self._load_metadata()
        new_or_updated_docs = []
        all_current_docs = []
        data_dir = self.document_loader.data_dir  # 从 loader 获取数据目录

        if force_rebuild:
            logger.warning("Force rebuild is enabled. All existing data will be wiped and rebuilt.")
            processed_files_meta.clear()

        try:
            # 1. 检测文件变更并加载文档
            # 【重构】现在 DocumentLoader 知道自己的 glob_pattern
            all_files_in_source = list(data_dir.rglob(self.document_loader.config.glob_pattern))
            for file_path in all_files_in_source:
                if not file_path.is_file(): continue

                file_rel_path = str(file_path.relative_to(data_dir))
                file_hash = self._get_file_hash(file_path)

                # 使用我们新加的 load_single_file 方法加载所有文档，用于BM25
                # 这是因为 BM25 通常需要全量数据来构建索引
                docs_from_file = self.document_loader.load_single_file(file_path)
                all_current_docs.extend(docs_from_file)

                # 仅当文件是新的或已更新时，才将其添加到待处理列表以更新向量库
                if processed_files_meta.get(file_rel_path) != file_hash or force_rebuild:
                    logger.info(f"Detected change in '{file_rel_path}'. Processing for vector update...")
                    new_or_updated_docs.extend(docs_from_file)
                    processed_files_meta[file_rel_path] = file_hash

            if not new_or_updated_docs and not force_rebuild:
                logger.info("No new or updated documents found. Ingestion process finished.")
                return

            # 2. 分割变更的文档
            chunks = self.text_splitter.split_documents(new_or_updated_docs)
            logger.info(f"Split {len(new_or_updated_docs)} documents into {len(chunks)} chunks.")

            # 3. 更新向量库
            if chunks:
                # 【修复逻辑】
                # 如果是强制重建，或者向量库尚未初始化（比如第一次运行），则必须从头构建
                if force_rebuild or not self.vector_store.is_initialized:
                    logger.info("Building vector store from scratch (Force rebuild or first run)...")
                    self.vector_store.build_from_documents(chunks)
                else:
                    # 只有在向量库已经存在且初始化成功的情况下，才进行增量添加
                    logger.info("Adding new chunks to existing vector store...")
                    self.vector_store.add_documents(chunks)

                logger.info("Vector store has been updated.")

            # 4. (重新)构建BM25索引 (如果配置中需要)
            if self.retrieval_config.retriever.strategy in ["bm25", "hybrid"]:
                logger.info(f"Rebuilding BM25 index with {len(all_current_docs)} total documents...")
                # 【重构】创建BM25实例并调用其构建方法
                bm25_retriever = BM25Retriever(self.retrieval_config.retriever.bm25_config, self.persist_dir)
                bm25_retriever.build_and_save(all_current_docs)
                logger.info("BM25 index has been rebuilt and saved.")

            # 5. 保存元数据
            self._save_metadata(processed_files_meta)
            logger.info("Data ingestion process completed successfully.")

        except Exception as e:
            raise DataProcessingError("Ingestion process failed.", original_exception=e)
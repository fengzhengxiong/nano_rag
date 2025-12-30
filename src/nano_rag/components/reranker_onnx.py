#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：reranker_onnx.py
@Author  ：fengzhengxiong
@Date    ：2025/12/30 09:50 
'''

import logging
import asyncio
import torch
from typing import List, Tuple

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from langsmith import traceable

from langchain_core.documents import Document
from ..core.interfaces import RerankerInterface
from ..core.exceptions import InitializationError, RetrievalError
from ..config.models import BGERerankerConfig

logger = logging.getLogger(__name__)


class ONNXBGEReranker(RerankerInterface):
    """
    基于 ONNX Runtime 的高性能重排序实现 (INT8 Quantized)。
    """

    def __init__(self, config: BGERerankerConfig):
        self._config = config
        self._model = None
        self._tokenizer = None

        try:
            logger.info(f"🚀 Initializing ONNX Reranker from: {config.model_name}")

            # 1. 加载 Tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)

            # 2. 加载 ONNX 模型 (自动寻找 model_quantized.onnx)
            # 这里的 file_name 必须对应转换脚本里生成的文件名
            self._model = ORTModelForSequenceClassification.from_pretrained(
                config.model_name,
                file_name="model_quantized.onnx"
            )

            logger.info("✅ ONNX Reranker initialized successfully (Backend: ONNX Runtime).")

        except Exception as e:
            raise InitializationError("ONNXBGEReranker", f"Failed to load ONNX model from {config.model_name}", e)

    @property
    def config(self) -> BGERerankerConfig:
        return self._config

    @traceable(name="BGE Reranker", run_type="retriever")  # 【新增这行】
    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """同步推理 (CPU INT8)"""
        if not documents: return []

        try:
            # 1. 构造输入对
            pairs = [[query, doc.page_content] for doc in documents]

            # 2. Tokenize (转为 Tensor)
            # truncation=True, max_length=512 限制长度
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # 3. 推理 (ONNX Runtime)
            # outputs.logits 形状为 [batch_size, 1]
            with torch.no_grad():
                outputs = self._model(**inputs)

            # 4. 提取分数 (Sigmoid 处理)
            logits = outputs.logits
            if logits.shape[1] == 1:
                scores = torch.sigmoid(logits).view(-1).tolist()
            else:
                # 某些模型可能输出 [batch, 2]，取正类分数
                scores = torch.softmax(logits, dim=1)[:, 1].tolist()

            # 5. 排序
            results = list(zip(documents, scores))
            results.sort(key=lambda x: x[1], reverse=True)

            return results

        except Exception as e:
            raise RetrievalError("ONNX Reranking process failed", e)

    async def arerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """异步包装"""
        if not documents: return []
        try:
            return await asyncio.to_thread(self.rerank, query, documents)
        except Exception as e:
            raise RetrievalError("Async ONNX reranking failed", e)
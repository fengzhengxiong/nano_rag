#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：query_models.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:05 
'''

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

class SourceDocument(BaseModel):
    """
    用于API响应的源文档模型，避免直接暴露Langchain内部对象。
    """
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

    @classmethod
    def from_langchain_doc(cls, lc_doc: Document, score: Optional[float] = None) -> 'SourceDocument':
        """从 Langchain Document 对象创建 SourceDocument 实例。"""
        return cls(
            page_content=lc_doc.page_content,
            metadata=lc_doc.metadata,
            score=score
        )

class QueryResponse(BaseModel):
    """
    标准化的查询响应模型。
    """
    answer: str
    source_documents: List[SourceDocument] = Field(default_factory=list)
    session_id: Optional[str] = None
    error: Optional[str] = None
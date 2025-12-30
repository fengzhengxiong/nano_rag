#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：schemas.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:08 
'''

from typing import List, Optional
from pydantic import BaseModel, Field

# --- Request Models (请求) ---

class ChatRequest(BaseModel):
    query: str = Field(..., description="用户的问题", example="Transformer 的优势是什么？")
    session_id: str = Field(default="default_session", description="会话ID，用于记忆上下文")
    stream: bool = Field(default=False, description="是否开启流式输出 (目前 V1.2 暂只支持 False)")

class IngestRequest(BaseModel):
    force_rebuild: bool = Field(default=False, description="是否强制重建索引")

# --- Response Models (响应) ---

class SourceDocumentDTO(BaseModel):
    """用于 API 返回的源文档精简结构"""
    source: str = Field(..., description="文件名")
    score: float = Field(..., description="相关性得分")
    content: str = Field(..., description="文档片段内容")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="AI 的回答")
    session_id: str
    sources: List[SourceDocumentDTO] = Field(default=[], description="引用来源")
    error: Optional[str] = None

class IngestResponse(BaseModel):
    status: str
    message: str
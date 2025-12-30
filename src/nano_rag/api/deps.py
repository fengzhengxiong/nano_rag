#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：deps.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:08 
'''

from typing import Optional
from fastapi import Request

from ..application import RAGApplication
from ..config.loader import get_resolved_config

# 全局单例变量
_app_instance: Optional[RAGApplication] = None

def initialize_global_application():
    """
    在服务器启动时调用：一次性加载配置和模型。
    """
    global _app_instance
    if _app_instance is None:
        print("🚀 [FastAPI] Initializing RAG Engine...")
        config = get_resolved_config()
        _app_instance = RAGApplication(config)
        print("✅ [FastAPI] RAG Engine ready.")

def get_rag_application() -> RAGApplication:
    """
    依赖注入函数：在 API 路由中获取 RAG 实例。
    如果服务器还没启动好就调用，会抛错。
    """
    if _app_instance is None:
        raise RuntimeError("RAG Application is not initialized!")
    return _app_instance

def get_query_service(request: Request):
    """
    获取 QueryService 的依赖函数。
    """
    app = get_rag_application()
    if not app.query_service:
        # 这里可以抛出一个 HTTP 503 Service Unavailable
        raise RuntimeError("Query Service is not ready (Maybe Ingestion needed?)")
    return app.query_service

def get_ingestion_service():
    """获取 IngestionService"""
    app = get_rag_application()
    return app.ingestion_service
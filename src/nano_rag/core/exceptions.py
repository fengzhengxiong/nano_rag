#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：exceptions.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:05 
'''

from typing import Optional

class RAGException(Exception):
    """RAG 系统所有自定义异常的基类。"""
    pass

class ConfigurationError(RAGException):
    """表示应用程序的配置存在错误。"""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        full_message = f"Configuration Error: {message}"
        if original_exception:
            full_message += f"\n  - Original Cause: {original_exception}"
        super().__init__(full_message)

class InitializationError(RAGException):
    """表示在组件初始化期间发生故障。"""
    def __init__(self, component_name: str, message: str, original_exception: Optional[Exception] = None):
        full_message = f"Failed to initialize component '{component_name}': {message}"
        if original_exception:
            full_message += f"\n  - Original Cause: {original_exception}"
        super().__init__(full_message)

class DataProcessingError(RAGException):
    """表示在数据提取或处理过程中发生错误。"""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        full_message = f"Data processing failed: {message}"
        if original_exception:
            full_message += f"\n  - Original Cause: {original_exception}"
        super().__init__(full_message)

class RetrievalError(RAGException):
    """表示在文档检索或重排期间发生错误。"""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        full_message = f"Retrieval failed: {message}"
        if original_exception:
            full_message += f"\n  - Original Cause: {original_exception}"
        super().__init__(full_message)

class GenerationError(RAGException):
    """表示在LLM生成答案期间发生错误。"""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        full_message = f"Answer generation failed: {message}"
        if original_exception:
            full_message += f"\n  - Original Cause: {original_exception}"
        super().__init__(full_message)
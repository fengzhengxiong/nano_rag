#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：database.py
@Author  ：fengzhengxiong
@Date    ：2025/12/30 11:16 
'''

import logging
from typing import Optional
from sqlmodel import Field, SQLModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# --- 1. 定义数据表模型 (Schema) ---
class ChatMessage(SQLModel, table=True):
    """
    聊天记录表
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True) # 加索引，方便按会话查询
    role: str       # "user" 或 "ai"
    content: str    # 聊天内容
    timestamp: float # 时间戳，用于排序

# --- 2. 配置数据库连接 ---
# 数据库文件将生成在项目根目录: chat_history.db
SQLITE_FILE_NAME = "chat_history.db"
SQLITE_URL = f"sqlite+aiosqlite:///{SQLITE_FILE_NAME}"

# 创建异步引擎
engine = create_async_engine(SQLITE_URL, echo=False)

# 创建异步 Session 工厂 (供 Service 使用)
AsyncSessionFactory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# --- 3. 初始化工具 ---
async def init_db():
    """
    在应用启动时调用，如果表不存在则创建。
    """
    logger.info(f"Checking database at: {SQLITE_FILE_NAME}")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Database initialized successfully.")

# 【新增】关闭数据库连接的方法
async def close_db():
    """
    在应用关闭时调用，释放数据库连接池。
    """
    logger.info("Closing database connection...")
    await engine.dispose()
    logger.info("Database connection closed.")
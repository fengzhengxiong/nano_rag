#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：history_service.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:02 
'''

# import logging
# from typing import List, Deque
# from collections import deque
#
# # 【升级】引入消息对象
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
#
# logger = logging.getLogger(__name__)
#
#
# class HistoryService:
#     """
#     会话历史管理器。
#     升级点：存储 BaseMessage 对象，适配 ChatModel 的输入格式。
#     """
#
#     def __init__(self, max_history_len: int = 5):
#         # 结构: {session_id: deque([HumanMessage, AIMessage, ...])}
#         # max_history_len 指的是“轮数”，但在 deque 里我们存的是消息数，所以要 * 2
#         self._storage: dict[str, Deque[BaseMessage]] = {}
#         self.max_history_len = max_history_len
#
#     async def add_turn(self, session_id: str, user_query: str, ai_answer: str):
#         """
#         (Async) 添加一轮对话。
#         预留 async 关键字，方便未来无缝迁移到 Redis/Postgres 存储。
#         """
#         if session_id not in self._storage:
#             self._storage[session_id] = deque(maxlen=self.max_history_len * 2)
#
#         # 存入标准消息对象
#         self._storage[session_id].append(HumanMessage(content=user_query))
#         self._storage[session_id].append(AIMessage(content=ai_answer))
#
#         logger.debug(f"Added turn to session '{session_id}'. Current msgs: {len(self._storage[session_id])}")
#
#     async def get_history_messages(self, session_id: str) -> List[BaseMessage]:
#         """
#         (Async) 获取 LangChain 格式的消息列表。
#         """
#         if session_id not in self._storage:
#             return []
#         return list(self._storage[session_id])
#
#     def get_history_as_text(self, session_id: str) -> str:
#         """
#         (兼容性接口) 获取文本格式历史，用于调试或旧式 Log。
#         """
#         msgs = self._storage.get(session_id, [])
#         return "\n".join([f"{m.type}: {m.content}" for m in msgs])
#
#     def clear(self, session_id: str):
#         if session_id in self._storage:
#             del self._storage[session_id]




import time
import logging
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from sqlmodel import select

# 导入刚才定义的 DB 组件
from ..core.database import ChatMessage, AsyncSessionFactory

logger = logging.getLogger(__name__)


class HistoryService:
    """
    基于 SQLite 的持久化历史记录服务。
    不再存储于内存，而是读写本地数据库文件。
    """

    def __init__(self, max_history_len: int = 5):
        self.max_history_len = max_history_len

    async def add_turn(self, session_id: str, user_query: str, ai_answer: str):
        """
        (Async) 将一问一答写入数据库。
        """
        now = time.time()

        # 创建记录对象
        # 稍微错开一点时间戳，确保 user 在前，ai 在后
        user_msg = ChatMessage(session_id=session_id, role="user", content=user_query, timestamp=now)
        ai_msg = ChatMessage(session_id=session_id, role="ai", content=ai_answer, timestamp=now + 0.001)

        try:
            # 打开一个短暂的会话进行写入
            async with AsyncSessionFactory() as session:
                session.add(user_msg)
                session.add(ai_msg)
                await session.commit()
                # 刷新以获取 ID (虽然这里不需要，但为了严谨)
                # await session.refresh(user_msg)
            logger.debug(f"Saved chat turn to DB for session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to save history to DB: {e}")

    async def get_history_messages(self, session_id: str) -> List[BaseMessage]:
        """
        (Async) 从数据库读取最近 N 轮对话，并转换为 LangChain 消息对象。
        """
        try:
            async with AsyncSessionFactory() as session:
                # 1. 查询逻辑：按时间倒序查，限制数量
                # limit = max_history_len * 2 (因为一轮包含 2 条消息)
                limit_count = self.max_history_len * 2

                statement = (
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.timestamp.desc())
                    .limit(limit_count)
                )

                result = await session.execute(statement)
                db_msgs = result.scalars().all()

                # 2. 因为是倒序查出来的(最新的在前)，需要反转回正常语序
                db_msgs = list(reversed(db_msgs))

                # 3. 转换为 LangChain 对象
                lc_msgs = []
                for msg in db_msgs:
                    if msg.role == "user":
                        lc_msgs.append(HumanMessage(content=msg.content))
                    elif msg.role == "ai":
                        lc_msgs.append(AIMessage(content=msg.content))

                return lc_msgs

        except Exception as e:
            logger.error(f"Failed to load history from DB: {e}")
            return []

    async def clear(self, session_id: str):
        """清空某个会话的历史"""
        try:
            async with AsyncSessionFactory() as session:
                # 这是一个删除操作
                # SQLModel/SQLAlchemy 的 delete 写法略有不同，这里用 execute
                from sqlalchemy import delete
                statement = delete(ChatMessage).where(ChatMessage.session_id == session_id)
                await session.execute(statement)
                await session.commit()
            logger.info(f"Cleared history for session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
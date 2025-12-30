#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：history_service.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:02 
'''

import logging
from typing import List, Deque
from collections import deque

# 【升级】引入消息对象
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class HistoryService:
    """
    会话历史管理器。
    升级点：存储 BaseMessage 对象，适配 ChatModel 的输入格式。
    """

    def __init__(self, max_history_len: int = 5):
        # 结构: {session_id: deque([HumanMessage, AIMessage, ...])}
        # max_history_len 指的是“轮数”，但在 deque 里我们存的是消息数，所以要 * 2
        self._storage: dict[str, Deque[BaseMessage]] = {}
        self.max_history_len = max_history_len

    async def add_turn(self, session_id: str, user_query: str, ai_answer: str):
        """
        (Async) 添加一轮对话。
        预留 async 关键字，方便未来无缝迁移到 Redis/Postgres 存储。
        """
        if session_id not in self._storage:
            self._storage[session_id] = deque(maxlen=self.max_history_len * 2)

        # 存入标准消息对象
        self._storage[session_id].append(HumanMessage(content=user_query))
        self._storage[session_id].append(AIMessage(content=ai_answer))

        logger.debug(f"Added turn to session '{session_id}'. Current msgs: {len(self._storage[session_id])}")

    async def get_history_messages(self, session_id: str) -> List[BaseMessage]:
        """
        (Async) 获取 LangChain 格式的消息列表。
        """
        if session_id not in self._storage:
            return []
        return list(self._storage[session_id])

    def get_history_as_text(self, session_id: str) -> str:
        """
        (兼容性接口) 获取文本格式历史，用于调试或旧式 Log。
        """
        msgs = self._storage.get(session_id, [])
        return "\n".join([f"{m.type}: {m.content}" for m in msgs])

    def clear(self, session_id: str):
        if session_id in self._storage:
            del self._storage[session_id]
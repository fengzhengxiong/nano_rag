#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：query_service.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:03 
'''

import logging
from typing import Optional, List, Any, AsyncIterator, Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from ..core.interfaces import LLMInterface, RetrieverInterface, RerankerInterface
from ..core.query_models import QueryResponse, SourceDocument
from .history_service import HistoryService

logger = logging.getLogger(__name__)


class QueryService:
    """
    升级版查询服务 (支持 Async LCEL & Streaming)。
    包含 ask (一次性返回) 和 ask_stream (流式生成) 两个核心入口。
    """

    def __init__(
            self,
            llm: LLMInterface,
            retriever: RetrieverInterface,
            reranker: Optional[RerankerInterface] = None
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.history_service = HistoryService(max_history_len=5)

        # 定义 Chain
        self.condense_q_chain: Optional[RunnableSerializable] = None
        self.qa_chain: Optional[RunnableSerializable] = None
        self.qa_prompt: Optional[ChatPromptTemplate] = None

        self._setup_chains()
        logger.info("QueryService initialized with Async LCEL & Streaming support.")

    def _setup_chains(self):
        llm = self.llm.get_langchain_llm()
        output_parser = StrOutputParser()

        # 1. 改写 Chain
        condense_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "给定以下对话历史和后续问题，请将后续问题改写为一个独立的、包含所有必要上下文的完整问题。\n如果后续问题本身已经是独立的，则保持原样。\n不要回答问题，只负责改写。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        self.condense_q_chain = condense_prompt | llm | output_parser

        # 2. QA Prompt (保存 Prompt 对象以便流式调用时手动构建 Chain)
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个专业的知识库助手。请基于以下提供的【上下文片段】回答用户的问题。\n\n上下文片段:\n{context}\n\n要求：\n1. 如果无法从上下文中得到答案，请明确告知。\n2. 回答要条理清晰，可以使用 Markdown 格式。"),
            ("human", "{question}")
        ])
        # 3. QA Chain (供非流式 ask 使用)
        self.qa_chain = self.qa_prompt | llm | output_parser

    async def _rewrite_query(self, query: str, history_msgs: List[Any]) -> str:
        """(Async) 使用 LLM 改写问题"""
        if not history_msgs:
            return query

        try:
            rewritten = await self.condense_q_chain.ainvoke({
                "chat_history": history_msgs,
                "question": query
            })
            rewritten = rewritten.strip()
            logger.info(f"Original: '{query}' -> Rewritten: '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}. Using original.")
            return query

    async def _retrieve_and_rerank(self, query: str) -> List[SourceDocument]:
        """(Internal) 封装检索和重排序逻辑，返回 SourceDocument 对象列表"""
        # 1. 检索
        retrieved_docs = await self.retriever.aretrieve(query)
        logger.info(f"Retrieved {len(retrieved_docs)} docs.")

        # 2. 重排序
        final_docs_with_scores = []
        if self.reranker and retrieved_docs:
            reranked = await self.reranker.arerank(query, retrieved_docs)
            final_docs_with_scores = reranked[:self.reranker.config.top_k]
        else:
            final_docs_with_scores = [(doc, 1.0) for doc in retrieved_docs]

        # 转换为 SourceDocument 模型
        return [
            SourceDocument.from_langchain_doc(doc, score)
            for doc, score in final_docs_with_scores
        ]

    async def ask(self, query: str, session_id: str) -> QueryResponse:
        """
        [普通模式] 一次性返回结果 (供 CLI 使用)。
        """
        try:
            # 1. 历史 & 改写
            history_msgs = await self.history_service.get_history_messages(session_id)
            search_query = await self._rewrite_query(query, history_msgs)

            # 2. 检索 & 重排
            source_docs = await self._retrieve_and_rerank(search_query)

            # 3. 构建 Context
            context_text = "\n\n".join([d.page_content for d in source_docs])

            # 4. 生成
            answer = await self.qa_chain.ainvoke({
                "context": context_text,
                "question": search_query
            })

            # 5. 存历史
            await self.history_service.add_turn(session_id, query, answer)

            return QueryResponse(
                session_id=session_id,
                answer=answer,
                source_documents=source_docs
            )
        except Exception as e:
            logger.error(f"Error in ask: {e}", exc_info=True)
            return QueryResponse(session_id=session_id, answer="Server Error", error=str(e))

    async def ask_stream(self, query: str, session_id: str) -> AsyncIterator[Dict[str, Any]]:
        """
        [流式模式] 生成器，依次返回状态、源文档、Token。
        供 Web API SSE 使用。
        Yields:
            dict: {"type": "status"|"sources"|"token"|"error", "content": ...}
        """
        logger.info(f"Starting STREAM query for session '{session_id}'")
        full_answer = ""  # 用于最后存历史

        try:
            # --- Step 1: 历史记录 ---
            # yield {"type": "status", "content": "正在读取记忆..."}
            history_msgs = await self.history_service.get_history_messages(session_id)

            # --- Step 2: 问题改写 ---
            yield {"type": "status", "content": "正在理解问题..."}
            search_query = await self._rewrite_query(query, history_msgs)

            # --- Step 3: 检索与重排 ---
            yield {"type": "status", "content": "正在搜索知识库..."}
            source_docs = await self._retrieve_and_rerank(search_query)

            # 立即返回源文档信息给前端
            # 将 Pydantic 对象转为 dict
            docs_json = [doc.model_dump() for doc in source_docs]
            yield {"type": "sources", "content": docs_json}

            # --- Step 4: 流式生成 ---
            yield {"type": "status", "content": "正在思考..."}
            context_text = "\n\n".join([d.page_content for d in source_docs])

            # 手动构建 Chain 以便使用 astream
            chain = self.qa_prompt | self.llm.get_langchain_llm() | StrOutputParser()

            async for token in chain.astream({
                "context": context_text,
                "question": search_query
            }):
                full_answer += token
                # 实时推送 Token
                yield {"type": "token", "content": token}

            # --- Step 5: 存历史 (后台完成) ---
            if full_answer:
                await self.history_service.add_turn(session_id, query, full_answer)
                logger.info(f"Stream finished. Answer length: {len(full_answer)}")

        except Exception as e:
            logger.error(f"Error in ask_stream: {e}", exc_info=True)
            yield {"type": "error", "content": str(e)}
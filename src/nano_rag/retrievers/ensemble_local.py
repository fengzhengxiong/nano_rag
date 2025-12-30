#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：ensemble_local.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:04 
'''

from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document


class EnsembleRetriever(BaseRetriever):
    """
    [本地实现] EnsembleRetriever
    由于 Python 3.14 环境下 langchain 主包导入不稳定，我们直接将核心逻辑内嵌到项目中。
    """
    retrievers: List[BaseRetriever]
    weights: List[float]

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        核心逻辑：加权倒数排名融合 (Weighted Reciprocal Rank Fusion) 的简化版
        或者直接使用加权分数融合。这里复刻 LangChain 标准的加权逻辑。
        """

        # 1. 获取所有检索器的结果
        all_docs_by_retriever = []
        for i, retriever in enumerate(self.retrievers):
            try:
                docs = retriever.invoke(query, config={"callbacks": run_manager.get_child()})
                all_docs_by_retriever.append(docs)
            except Exception:
                # 容错：如果某个检索器挂了，不要通过崩溃整个流程，而是视为空结果
                all_docs_by_retriever.append([])

        # 2. 简单的加权融合算法 (Weighted Rank Fusion)
        # 统计每个文档在不同检索器中的排名权重
        doc_scores = {}

        for retriever_idx, docs in enumerate(all_docs_by_retriever):
            current_weight = self.weights[retriever_idx]

            for rank, doc in enumerate(docs):
                # 这里的 key 使用 page_content，注意：如果内容完全一样会被去重
                # 更严谨的做法是 hash 内容 + metadata source
                doc_key = doc.page_content

                # 算法：分数 = 权重 * (1 / (排名 + 1))
                # 排名越靠前 (rank越小)，分数越高
                score = current_weight * (1.0 / (rank + 1))

                if doc_key in doc_scores:
                    doc_scores[doc_key]["score"] += score
                else:
                    doc_scores[doc_key] = {
                        "score": score,
                        "doc": doc
                    }

        # 3. 排序并输出
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [item["doc"] for item in sorted_docs]

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        异步获取文档。并行执行所有子检索器。
        """
        import asyncio

        # 1. 创建所有子检索器的异步任务
        tasks = []
        for retriever in self.retrievers:
            # 调用子检索器的 ainvoke
            tasks.append(retriever.ainvoke(query, config={"callbacks": run_manager.get_child()}))

        # 2. 并行等待所有结果 (Gather)
        # return_exceptions=True 防止一个检索器挂了影响其他
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. 清洗结果 (去除异常)
        valid_results = []
        for res in results:
            if isinstance(res, list):
                valid_results.append(res)
            else:
                valid_results.append([])  # 发生异常当作空结果

        # 4. 复用同步的融合逻辑 (因为融合是纯 CPU 计算，且很快，不需要再 async)
        # 这里为了复用逻辑，我们手动构造一个假的 "all_docs_by_retriever" 传给同步逻辑？
        # 或者把 _get_relevant_documents 拆分成 "fetch" 和 "rank" 两步。
        # 为了简单，直接复制一下 rank 逻辑吧，或者简单的：

        # --- 复制 Ranking 逻辑 Start ---
        doc_scores = {}
        for retriever_idx, docs in enumerate(valid_results):
            current_weight = self.weights[retriever_idx]
            for rank, doc in enumerate(docs):
                doc_key = doc.page_content
                score = current_weight * (1.0 / (rank + 1))
                if doc_key in doc_scores:
                    doc_scores[doc_key]["score"] += score
                else:
                    doc_scores[doc_key] = {"score": score, "doc": doc}

        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]
        # --- 复制 Ranking 逻辑 End ---
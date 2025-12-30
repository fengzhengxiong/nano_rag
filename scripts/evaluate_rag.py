#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：evaluate_rag.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 16:57 
'''

import os
import sys
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nano_rag.config.loader import get_resolved_config
from src.nano_rag.application import RAGApplication

# 1. 配置评委模型 (Judge Model)
# 我们使用 DeepSeek V3 作为裁判，它足够聪明且便宜
config = get_resolved_config()
judge_llm = ChatOpenAI(
    base_url=config.llm.base_url,
    api_key=config.llm.api_key,
    model=config.llm.model_name,
    temperature=0
)

# 2. 配置 Embedding 模型 (用于评估相似度)
# 复用本地的 BGE 模型
embedding_model = HuggingFaceEmbeddings(
    model_name=config.embedding.model_name,
    encode_kwargs={'normalize_embeddings': True}
)


def prepare_test_data(app: RAGApplication):
    """
    准备测试数据集 (Golden Dataset)。
    """
    questions = [
        "FZX 集团 Q4 企业级 RAG 一体机的营收是多少？",
        "哪个产品线的毛利率最高？",
        "核心架构部主要在哪里办公？",
        "数据清洗服务的环比增长是多少？",
    ]

    # 【修复点】这里必须是字符串列表，不能是列表的列表
    # 旧写法: [["180.0 百万元"], ...]  <-- 报错原因
    # 新写法: ["180.0 百万元", ...]
    ground_truths = [
        "180.0 百万元",
        "AI 安全网关，毛利率为 72%",
        "北京",
        "-5.0%",
    ]

    answers = []
    contexts = []

    print("🚀 开始运行 RAG 系统生成答案...")

    import asyncio

    async def run_queries():
        for q in questions:
            print(f"Querying: {q} ...")
            resp = await app.query_service.ask(q, session_id="eval_bot")
            answers.append(resp.answer)
            # 提取召回的上下文内容
            ctx_list = [doc.page_content for doc in resp.source_documents]
            contexts.append(ctx_list)

    asyncio.run(run_queries())

    # 构建 Ragas 所需的数据集格式
    # Ragas v0.2+ 会自动将 ground_truth 映射为 reference
    data = {
        "user_input": questions,  # 新版建议用 user_input 而不是 question
        "response": answers,  # 新版建议用 response 而不是 answer
        "retrieved_contexts": contexts,  # 新版建议用 retrieved_contexts
        "reference": ground_truths  # 新版建议用 reference
    }
    return Dataset.from_dict(data)


def main():
    print("🔄 初始化 RAG 应用...")
    app = RAGApplication(config)

    print("🛠️ 准备测试数据...")
    dataset = prepare_test_data(app)

    print("⚖️ 开始 Ragas 评估 (这可能需要几分钟)...")
    # 这一步会调用 Judge LLM 对每一条问答进行打分
    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,  # 检索精度：检索到的内容里有多少是有用的？
            context_recall,  # 检索召回：标准答案需要的信息都查到了吗？
            faithfulness,  # 忠实度：答案是否完全基于上下文（没幻觉）？
            answer_relevancy,  # 相关性：答非所问了吗？
        ],
        llm=judge_llm,
        embeddings=embedding_model
    )

    print("\n" + "=" * 50)
    print("📊 评估报告 (Evaluation Report)")
    print("=" * 50)
    print(results)

    # 导出为 Excel 方便给老板看
    df = results.to_pandas()
    output_file = "evaluation_report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n✅ 详细报告已保存至: {output_file}")


if __name__ == "__main__":
    main()
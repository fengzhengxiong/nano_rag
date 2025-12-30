#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：main.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:08 
'''

# import time
# import logging
# from contextlib import asynccontextmanager
# from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from pathlib import Path
#
# # 导入我们定义的模块
# from .schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse, SourceDocumentDTO
# from .deps import initialize_global_application, get_query_service, get_ingestion_service
# from ..services.query_service import QueryService
# from ..services.ingestion_service import IngestionService
# from ..core.query_models import QueryResponse
#
# # 设置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("fastapi_app")
#
#
# # --- 生命周期管理 ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     FastAPI 官方推荐的生命周期管理方式。
#     启动前：加载模型。
#     关闭后：清理资源（如果有）。
#     """
#     try:
#         # 1. 初始化 RAG 引擎 (加载 LLM, Embedding, VectorStore)
#         initialize_global_application()
#         yield
#     except Exception as e:
#         logger.error(f"Failed to start application: {e}")
#         raise
#     finally:
#         logger.info("Application shutting down...")
#
#
# # --- 初始化 App ---
# app = FastAPI(
#     title="NANO-RAG Enterprise API",
#     version="1.2.0",
#     description="基于混合云架构的企业级知识库问答系统",
#     lifespan=lifespan
# )
#
# # --- CORS 跨域设置 (允许前端访问) ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 生产环境建议改为具体的 ["http://localhost:3000"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# # --- API 路由 ---
#
# @app.get("/health")
# async def health_check():
#     """健康检查接口"""
#     return {"status": "ok", "version": "1.2.0"}
#
#
# @app.post("/api/v1/chat", response_model=ChatResponse)
# async def chat_endpoint(
#         request: ChatRequest,
#         service: QueryService = Depends(get_query_service)
# ):
#     """
#     对话接口 (V1.2 同步等待模式)
#     未来 V1.3 将在此增加 StreamingResponse 支持
#     """
#     start_time = time.time()
#     logger.info(f"Received chat request: {request.query} (Session: {request.session_id})")
#
#     try:
#         # 调用核心业务 (全链路异步)
#         result: QueryResponse = await service.ask(
#             query=request.query,
#             session_id=request.session_id
#         )
#
#         if result.error:
#             raise HTTPException(status_code=500, detail=result.error)
#
#         # 转换 Domain Model -> API DTO
#         sources_dto = []
#         if result.source_documents:
#             for doc in result.source_documents:
#                 # 安全获取文件名
#                 src_name = Path(doc.metadata.get("source", "unknown")).name
#                 sources_dto.append(SourceDocumentDTO(
#                     source=src_name,
#                     score=doc.score if doc.score else 0.0,
#                     content=doc.page_content
#                 ))
#
#         elapsed = time.time() - start_time
#         logger.info(f"Request processed in {elapsed:.2f}s")
#
#         return ChatResponse(
#             answer=result.answer,
#             session_id=result.session_id or request.session_id,
#             sources=sources_dto
#         )
#
#     except Exception as e:
#         logger.error(f"Error processing chat request: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.post("/api/v1/ingest", response_model=IngestResponse)
# async def ingest_endpoint(
#         request: IngestRequest,
#         background_tasks: BackgroundTasks,
#         service: IngestionService = Depends(get_ingestion_service)
# ):
#     """
#     触发数据更新 (异步后台任务)。
#     因为 Ingestion 可能很慢，我们立即返回，并在后台运行。
#     """
#
#     def _run_ingest_job(force: bool):
#         logger.info("Starting background ingestion job...")
#         try:
#             service.run(force_rebuild=force)
#             logger.info("Background ingestion job finished.")
#         except Exception as e:
#             logger.error(f"Background ingestion failed: {e}")
#
#     # 添加到后台任务队列
#     background_tasks.add_task(_run_ingest_job, request.force_rebuild)
#
#     return IngestResponse(
#         status="accepted",
#         message="Data ingestion task has been started in the background."
#     )
#
# @app.post("/api/v1/chat/stream")
# async def chat_stream(request: ChatRequest):
#     async def event_generator():
#         # 调用 LLM 的 astream
#         async for token in service.ask_stream(request.query):
#             yield f"data: {token}\n\n"
#     return StreamingResponse(event_generator(), media_type="text/event-stream")
#
#
# if __name__ == "__main__":
#     import uvicorn
#
#     # 本地调试运行
#     uvicorn.run("src.nano_rag.api.main:app", host="0.0.0.0", port=8000, reload=True)


import time
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse  # 【新增】
from pathlib import Path

from .schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse, SourceDocumentDTO
from .deps import initialize_global_application, get_query_service, get_ingestion_service
from ..services.query_service import QueryService
from ..services.ingestion_service import IngestionService
from ..core.query_models import QueryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_app")


# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize_global_application()
        yield
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        logger.info("Application shutting down...")


app = FastAPI(
    title="RAG-FZX Enterprise API",
    version="1.3.0",  # 升级版本号
    description="支持 SSE 流式输出的企业级 RAG 系统",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API 路由 ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.3.0"}


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
        request: ChatRequest,
        service: QueryService = Depends(get_query_service)
):
    """(同步接口) 等待完全生成后返回"""
    start_time = time.time()
    try:
        result: QueryResponse = await service.ask(
            query=request.query,
            session_id=request.session_id
        )
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        sources_dto = []
        if result.source_documents:
            for doc in result.source_documents:
                src_name = Path(doc.metadata.get("source", "unknown")).name
                sources_dto.append(SourceDocumentDTO(
                    source=src_name,
                    score=doc.score if doc.score else 0.0,
                    content=doc.page_content
                ))

        elapsed = time.time() - start_time
        logger.info(f"Sync request processed in {elapsed:.2f}s")

        return ChatResponse(
            answer=result.answer,
            session_id=result.session_id or request.session_id,
            sources=sources_dto
        )
    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat/stream")
async def chat_stream_endpoint(
        request: ChatRequest,
        service: QueryService = Depends(get_query_service)
):
    """
    【新增】(流式接口) Server-Sent Events (SSE)
    前端可以通过 EventSource 或 fetch 流式读取。
    """
    logger.info(f"Stream request: {request.query}")

    async def event_generator():
        """将 Service 的生成器转换为 SSE 格式"""
        try:
            # 获取 QueryService 的异步生成器
            async for event_data in service.ask_stream(request.query, request.session_id):
                # event_data 是字典: {"type": "...", "content": ...}
                # SSE 标准格式: data: <json_string>\n\n
                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            err_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(err_data)}\n\n"

        # 结束标志 (可选，前端用来判断连接关闭)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_endpoint(
        request: IngestRequest,
        background_tasks: BackgroundTasks,
        service: IngestionService = Depends(get_ingestion_service)
):
    def _run_ingest_job(force: bool):
        logger.info("Starting background ingestion job...")
        try:
            service.run(force_rebuild=force)
            logger.info("Background ingestion job finished.")
        except Exception as e:
            logger.error(f"Background ingestion failed: {e}")

    background_tasks.add_task(_run_ingest_job, request.force_rebuild)
    return IngestResponse(status="accepted", message="Data ingestion started in background.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.rag_fzx.api.main:app", host="0.0.0.0", port=8000, reload=True)
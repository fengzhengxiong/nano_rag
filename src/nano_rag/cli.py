#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：cli.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:00 
'''

import sys
import logging
import argparse
import asyncio
import logging.handlers
from typing import NoReturn
from pathlib import Path

from .config.loader import get_resolved_config
from .config.models import LoggingConfig
from .core.exceptions import RAGException
from .application import RAGApplication


# ==============================================================================
# 日志设置 (保持不变)
# ==============================================================================
def setup_logging(config: LoggingConfig):
    """根据配置设置全局日志系统。"""
    log_dir = config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "nano_rag.log"

    log_level = getattr(logging, config.log_level.upper())

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    root_logger.info(f"Logging configured. Level: {config.log_level}, File: '{log_file}'")


# ==============================================================================
# 辅助函数
# ==============================================================================
def fatal_error(message: str) -> NoReturn:
    print(f"\nFATAL ERROR: {message}", file=sys.stderr)
    sys.exit(1)


# ==============================================================================
# 异步动作函数 (Async Actions)
# ==============================================================================

def run_ingest(app: RAGApplication, force_rebuild: bool):
    """
    执行数据注入流程。
    注：IngestionService 目前仍是同步的（CPU密集型操作为主），直接调用即可。
    """
    logger = logging.getLogger(__name__)
    logger.info("CLI action: Ingest")
    print("\nProcessing data ingestion...")
    # 直接调用同步方法
    app.ingestion_service.run(force_rebuild=force_rebuild)
    print("\n✅ Data ingestion finished successfully.")


async def run_ask(app: RAGApplication, query: str):
    """
    (Async) 处理单次问答。
    """
    logger = logging.getLogger(__name__)

    if app.query_service is None:
        fatal_error("QueryService is not initialized. Please run 'ingest' first to build the index.")

    if not query:
        fatal_error("The 'ask' action requires a --query (-q) argument.")

    logger.info(f"CLI action: Ask. Query: '{query}'")
    print("\nThinking...")

    # 【核心修改】使用 await 等待结果
    response = await app.query_service.ask(query=query, session_id="cli_ask_session")

    if response.error:
        fatal_error(f"An error occurred while processing your question: {response.error}")

    print("\n" + " Answer ".center(80, "─"))
    print(f"\n💡 {response.answer}")
    print("\n" + " Sources ".center(80, "─"))
    if response.source_documents:
        for i, doc in enumerate(response.source_documents, 1):
            score_str = f"{doc.score:.4f}" if doc.score is not None else "N/A"
            source = Path(doc.metadata.get('source', 'Unknown')).name
            print(f"\n[{i}] Source: {source} (Score: {score_str})")
            print("-" * 80)
            print(doc.page_content.strip())
    else:
        print("No source documents were retrieved for this answer.")
    print("\n" + "─" * 80)


async def run_chat(app: RAGApplication):
    """
    (Async) 启动交互式聊天会话。
    """
    logger = logging.getLogger(__name__)

    if app.query_service is None:
        fatal_error("QueryService is not initialized. Please run 'ingest' first.")

    logger.info("CLI action: Chat")
    print("\n🤖 Starting interactive chat session (Async). Type 'exit' or 'quit' to end.")
    session_id = "cli_chat_session"

    while True:
        try:
            # input() 是阻塞的，但在 CLI 这种单用户场景下没问题
            query = input("\n👤 You: ").strip()

            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("🤖 AI: Goodbye!")
                break

            # 【核心修改】使用 await
            response = await app.query_service.ask(query=query, session_id=session_id)

            if response.error:
                print(f"🤖 AI (error): {response.error}")
            else:
                print(f"🤖 AI: {response.answer}")

        except (KeyboardInterrupt, EOFError):
            print("\n🤖 AI: Session ended. Goodbye!")
            break


# ==============================================================================
# 主函数 (Async Entry Point)
# ==============================================================================
async def main_async():
    """异步主函数逻辑"""
    parser = argparse.ArgumentParser(description="A modular RAG system CLI (Async).")
    parser.add_argument("action", choices=["ingest", "ask", "chat"], help="The action to perform.")
    parser.add_argument("-q", "--query", type=str, help="The question to ask.")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of all data.")
    args = parser.parse_args()

    try:
        config = get_resolved_config()
        setup_logging(config.logging)

        # 初始化应用 (同步)
        app = RAGApplication(config)

        # 根据动作分发
        if args.action == "ingest":
            # 包装同步函数到 async 上下文中运行
            run_ingest(app, args.force_rebuild)
        elif args.action == "ask":
            await run_ask(app, args.query)
        elif args.action == "chat":
            await run_chat(app)

    except RAGException as e:
        fatal_error(str(e))
    except Exception as e:
        logging.getLogger(__name__).critical("An unexpected critical error occurred!", exc_info=True)
        fatal_error(f"An unexpected critical error occurred: {e}")


def main():
    """程序入口：启动 Event Loop"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

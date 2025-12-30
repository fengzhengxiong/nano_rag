#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：generate_requirements.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 17:22 
'''

import os

def generate_files():
    # ---------------------------------------------------------
    # 1. 定义 requirements.txt 内容
    # 策略：排除 PyTorch (避免自动下载 GPU 版)，包含所有 V1.3 新特性依赖
    # ---------------------------------------------------------
    req_content = """# ==========================================
# RAG-FZX Project Dependencies
# Generated for Python 3.11 (Recommended)
# ==========================================

# --- 1. RAG Core & Framework ---
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langchain-openai>=0.2.0        # LLM Connector
langchain-huggingface>=0.1.0   # Embedding Connector

# --- 2. Vector Store & Retrieval ---
faiss-cpu>=1.8.0               # Vector Database
sentence-transformers>=3.1.0   # Embedding Model
rank_bm25>=0.2.2               # Keyword Search

# --- 3. Web API & UI (V1.2/V1.3) ---
fastapi>=0.110.0               # Backend API
uvicorn[standard]>=0.29.0      # ASGI Server
streamlit>=1.35.0              # Frontend UI
httpx>=0.27.0                  # Async HTTP Client
requests>=2.31.0

# --- 4. Data Processing (ETL) ---
docling>=2.0.0                 # Deep PDF/Table Parsing
chardet>=5.2.0                 # Encoding Detection

# --- 5. Persistence & Database ---
sqlmodel>=0.0.16               # ORM (based on Pydantic/SQLAlchemy)
aiosqlite>=0.20.0              # Async SQLite Driver

# --- 6. Optimization (ONNX) ---
optimum[onnxruntime]>=1.17.0   # Model Quantization & Inference
onnxruntime>=1.17.0

# --- 7. Evaluation & Testing ---
ragas>=0.2.0                   # RAG Evaluation Framework
datasets>=2.19.0
pandas>=2.2.0
openpyxl>=3.1.0
reportlab>=4.2.0               # Test Data Generation

# --- 8. Infrastructure ---
pydantic>=2.9.0
pyyaml>=6.0
tiktoken>=0.7.0
"""

    # ---------------------------------------------------------
    # 2. 定义 INSTALL.md 内容 (分步安装指南)
    # ---------------------------------------------------------
    install_guide_content = """# 📦 RAG-FZX 安装与部署指南

为了确保 **Docling (PDF解析)** 和 **ONNX (模型加速)** 正常工作，请严格按照以下顺序安装依赖。

### ✅ 环境要求
*   **OS**: Windows / Linux / macOS
*   **Python**: 3.11 (强烈推荐，兼容性最佳)
*   **RAM**: 建议 8GB 以上 (运行本地大模型)

---

### 🚀 第一步：创建纯净环境
请不要在旧环境中混合安装，容易产生依赖冲突。

```bash
# 1. 创建环境
conda create -n rag_fzx python=3.11 -y

# 2. 激活环境
conda activate rag_fzx
🚀 第二步：优先安装 PyTorch (关键)
Docling 和 Embedding 模型强依赖 PyTorch。我们手动安装 CPU 版以减小体积（约 200MB）。
(方案 A: 普通电脑/笔记本 - 推荐)
code
Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
(方案 B: 有 NVIDIA 显卡 - 需要 GPU 加速)
code
Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
🚀 第三步：一键安装项目依赖
这一步会安装 LangChain, FastAPI, Docling, SQLModel 等所有组件。
code
Bash
pip install -r requirements.txt
☕ 提示: 这一步会自动下载 Docling 所需的 OCR 模型依赖，可能需要几分钟，请耐心等待。
🚀 第四步：环境自测
运行以下命令，如果没有报错，说明环境配置完美！
code
Bash
python -c "import torch; import docling; import sqlmodel; import optimum; print('✅ 恭喜！环境配置成功！')"
"""

    try:
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write(req_content)
        print("✅ [SUCCESS] 已生成依赖列表: requirements.txt")
    except Exception as e:
        print(f"❌ 生成 requirements.txt 失败: {e}")

    try:
        with open("INSTALL.md", "w", encoding="utf-8") as f:
            f.write(install_guide_content)
        print("✅ [SUCCESS] 已生成安装手册: INSTALL.md")
    except Exception as e:
        print(f"❌ 生成 INSTALL.md 失败: {e}")


if __name__ == "__main__":
    generate_files()


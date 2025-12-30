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
    # 策略：不包含 torch/torchvision，强制用户在 INSTALL.md 中
    # 根据自己的硬件（CPU vs GPU）手动选择安装命令，避免下载 3GB 的 CUDA 包。
    # ---------------------------------------------------------
    req_content = """# ==========================================
# RAG-FZX Project Dependencies
# Generated for Python 3.11 (Recommended)
# ==========================================

# --- 1. RAG Framework (LangChain Ecosystem) ---
# 锁定在 0.3.x 体系，确保稳定性
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langchain-openai>=0.2.0        # 用于连接 DeepSeek/SiliconFlow
langchain-huggingface>=0.1.0   # 用于 BGE Embedding

# --- 2. Retrieval & Vector Store ---
faiss-cpu>=1.8.0               # 向量数据库
sentence-transformers>=3.1.0   # 必须 >=3.0 以适配新版 BGE
rank_bm25>=0.2.2               # 混合检索算法

# --- 3. Web API & Frontend ---
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
streamlit>=1.35.0
httpx>=0.27.0                  # 异步 HTTP 请求库
requests>=2.31.0

# --- 4. Deep Document Parsing (ETL) ---
docling>=2.0.0                 # IBM 深度文档解析 (PDF/Table)
chardet>=5.2.0                 # 编码检测辅助

# --- 5. Evaluation & Testing ---
ragas>=0.2.0                   # RAG 评估框架
datasets>=2.19.0
pandas>=2.2.0
openpyxl>=3.1.0                # Excel 导出依赖
reportlab>=4.2.0               # 用于生成测试 PDF 数据

# --- 6. Utilities ---
pydantic>=2.9.0
pyyaml>=6.0
tiktoken>=0.7.0
"""

    # ---------------------------------------------------------
    # 2. 定义 INSTALL.md 内容 (分步安装指南)
    # ---------------------------------------------------------
    install_guide_content = """# 📦 RAG-FZX 安装指南 (Installation Guide)

为了确保依赖项正确安装（特别是 PyTorch 和 Docling 的兼容性），请**严格按照以下顺序**操作。

### ✅ 前置要求
*   **OS**: Windows / Linux / macOS
*   **Python**: 3.11 (强烈推荐，兼容性最佳)
*   **Conda**: 建议使用 Anaconda 或 Miniconda 管理环境

---

### 🚀 第一步：创建纯净环境
请不要在旧环境中混合安装，容易产生依赖冲突。

```bash
# 1. 创建名为 rag_fzx 的环境
conda create -n rag_fzx python=3.11 -y

# 2. 激活环境
conda activate rag_fzx
🚀 第二步：优先安装 PyTorch (关键)
docling 和 sentence-transformers 都强依赖 PyTorch。
我们建议手动安装，以便控制版本（CPU vs GPU）。
👉 方案 A：普通电脑 / 笔记本 (推荐 - CPU 版)
下载速度快 (约 200MB)，兼容性 100%，适合演示和开发。
code
Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
👉 方案 B：有 NVIDIA 显卡 (GPU 版)
如果你需要更快的推理速度，且网络环境良好 (需下载 2.5GB+)。
code
Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
🚀 第三步：安装项目依赖
这一步会安装 LangChain, FastAPI, Docling 等其余库。
code
Bash
pip install -r requirements.txt
☕ 提示: 这一步会自动下载 Docling 所需的 OCR 模型依赖，可能需要几分钟，请耐心等待。
🚀 第四步：环境自测
运行以下命令，如果没有报错，说明环境配置完美！
code
Bash
python -c "import torch; import docling; import fastapi; print('✅ 恭喜！环境配置成功！')"
"""

    # 3. 写入文件
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

    print("\n👉 完成！现在你可以将这两个文件随项目一起提交到 GitHub 了。")


if __name__ == "__main__":
    generate_files()
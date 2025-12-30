# 📦 RAG-FZX 安装与部署指南

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

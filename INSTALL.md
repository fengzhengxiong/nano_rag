# 📦 RAG-FZX 安装指南 (Installation Guide)

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

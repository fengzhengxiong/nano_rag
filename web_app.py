#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：web_app.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 11:35 
'''

import streamlit as st
import requests
import json
import time

# --- 全局配置 ---
API_BASE_URL = "http://127.0.0.1:8000/api/v1"  # 确保地址正确
PAGE_TITLE = "NANO-RAG Enterprise"
PAGE_ICON = "⚡️"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 CSS 深度美化 (关键部分) ---
st.markdown("""
<style>
    /* 1. 全局字体优化 */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* 2. 聊天气泡样式增强 */
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* 用户气泡: 淡蓝色背景，靠右视觉习惯(Streamlit默认靠左，这里通过颜色区分) */
    [data-testid="stChatMessage"][data-test="user"] {
        background-color: #E3F2FD;
        border: 1px solid #BBDEFB;
    }

    /* AI 气泡: 白色背景，灰色边框 */
    [data-testid="stChatMessage"][data-test="assistant"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
    }

    /* 3. 引用来源卡片样式 */
    .source-card {
        background-color: #F8F9FA;
        border-left: 4px solid #1f77b4; /* 蓝色左边条 */
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 8px;
        font-size: 0.9em;
        transition: all 0.2s;
    }
    .source-card:hover {
        background-color: #F1F3F5;
        transform: translateX(2px);
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }
    .source-filename {
        font-weight: 600;
        color: #2c3e50;
    }
    .source-score {
        background-color: #e6fcf5;
        color: #0ca678;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .source-content {
        color: #555;
        font-size: 0.9em;
        line-height: 1.4;
    }

    /* 4. 状态指示器样式 */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        background-color: #f1f3f5;
        border-radius: 20px;
        color: #495057;
        font-size: 0.85em;
        margin-bottom: 10px;
        border: 1px solid #dee2e6;
    }
    .blink {
        animation: blinker 1.5s linear infinite;
        color: #1f77b4;
        font-weight: bold;
        margin-right: 5px;
    }
    @keyframes blinker {
        50% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"user_{int(time.time())}"

# --- 侧边栏 ---
with st.sidebar:
    st.title(f"{PAGE_ICON} 知识库助手")
    st.caption("🚀 Powered by FastAPI & Asyncio")

    st.divider()

    st.subheader("📂 知识库管理")
    uploaded_files = st.file_uploader("上传文档 (模拟)", accept_multiple_files=True)

    if st.button("🔄 更新索引 (Ingest)", use_container_width=True):
        with st.status("正在处理数据...", expanded=True) as status:
            try:
                st.write("📤 正在连接后端服务...")
                resp = requests.post(f"{API_BASE_URL}/ingest", json={"force_rebuild": False}, timeout=5)
                if resp.status_code == 200:
                    status.update(label="✅ 索引更新任务已后台启动！", state="complete")
                else:
                    status.update(label="❌ 请求被拒绝", state="error")
                    st.error(resp.text)
            except Exception as e:
                status.update(label="❌ 连接失败", state="error")
                st.error(f"无法连接到 API: {e}")

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 主界面逻辑 ---

# 1. 渲染历史消息
for msg in st.session_state.messages:
    # data-test 属性用于 CSS 定位颜色
    role_attr = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(msg["role"]):
        # 渲染文本
        st.markdown(msg["content"])

        # 如果是 AI 回复且有源文档，渲染漂亮的卡片
        if msg.get("sources"):
            with st.expander(f"📚 参考文档 ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span class="source-filename">📄 {src['source']}</span>
                            <span class="source-score">{src.get('score', 0) * 100:.1f}% 相关</span>
                        </div>
                        <div class="source-content">{src['content']}...</div>
                    </div>
                    """, unsafe_allow_html=True)

# 2. 处理新输入
if prompt := st.chat_input("输入问题，例如：Transformer 的核心机制是什么？"):
    # 立即上屏用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 准备 AI 回答容器
    with st.chat_message("assistant"):
        # 占位符：用于显示动态状态 (检索中/思考中)
        status_placeholder = st.empty()
        # 占位符：用于流式显示答案
        answer_placeholder = st.empty()

        full_response = ""
        current_sources = []

        try:
            # 初始状态
            status_placeholder.markdown("""
                <div class="status-badge">
                    <span class="blink">●</span> 正在连接大脑...
                </div>
            """, unsafe_allow_html=True)

            with requests.post(
                    f"{API_BASE_URL}/chat/stream",
                    json={"query": prompt, "session_id": st.session_state.session_id},
                    stream=True,
                    timeout=60
            ) as response:

                if response.status_code != 200:
                    status_placeholder.empty()
                    st.error(f"Server Error: {response.status_code}")
                    st.code(response.text)
                else:
                    for line in response.iter_lines():
                        if not line: continue
                        line_text = line.decode('utf-8')
                        if not line_text.startswith("data: "): continue

                        data_str = line_text[6:]
                        if data_str == "[DONE]": break

                        try:
                            data = json.loads(data_str)
                            msg_type = data.get("type")
                            content = data.get("content")

                            # --- 状态机处理 ---
                            if msg_type == "status":
                                # 更新状态条
                                icon = "🔍" if "搜索" in content else "🧠"
                                status_placeholder.markdown(f"""
                                    <div class="status-badge">
                                        <span class="blink">●</span> {icon} {content}
                                    </div>
                                """, unsafe_allow_html=True)

                            elif msg_type == "sources":
                                # 收到源文档数据，处理一下并暂存
                                current_sources = content
                                for src in current_sources:
                                    path_str = src.get("metadata", {}).get("source", "unknown")
                                    # 提取文件名
                                    filename = path_str.replace("\\", "/").split("/")[-1]
                                    src["source"] = filename
                                    # 截断内容
                                    src["content"] = src.get("page_content", "")[:150].replace("\n", " ")

                            elif msg_type == "token":
                                # 收到正文，打字机输出
                                full_response += content
                                answer_placeholder.markdown(full_response + "▌")

                            elif msg_type == "error":
                                st.error(f"Error: {content}")

                        except json.JSONDecodeError:
                            pass

            # --- 完成后的收尾 ---
            # 1. 移除光标
            answer_placeholder.markdown(full_response)
            # 2. 移除状态条 (或者改成"完成"状态，这里选择移除保持清爽)
            status_placeholder.empty()

            # 3. 如果有源文档，渲染漂亮的折叠卡片
            if current_sources:
                with st.expander(f"📚 参考文档 ({len(current_sources)})", expanded=False):
                    for src in current_sources:
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">
                                <span class="source-filename">📄 {src['source']}</span>
                                <span class="source-score">{src.get('score', 0) * 100:.1f}% Match</span>
                            </div>
                            <div class="source-content">{src['content']}...</div>
                        </div>
                        """, unsafe_allow_html=True)

            # 4. 存入 Session 历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": current_sources
            })

        except Exception as e:
            status_placeholder.empty()
            st.error(f"连接失败: {str(e)}")
            st.warning("请检查 uvicorn 后端是否已在 8000 端口启动。")
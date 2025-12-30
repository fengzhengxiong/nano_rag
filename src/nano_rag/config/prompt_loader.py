#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：prompt_loader.py
@Author  ：fengzhengxiong
@Date    ：2025/12/30 11:44 
'''

import yaml
from pydantic import BaseModel
from ..utils.helpers import find_project_root


class PromptConfig(BaseModel):
    condense_q_system: str
    qa_system: str


def load_prompts(filename: str = "prompts.yaml") -> PromptConfig:
    """加载 Prompt 配置文件"""
    root = find_project_root()
    config_path = root / "configs" / filename

    if not config_path.exists():
        # 如果文件不存在，返回默认的兜底配置 (Fail-safe)
        return PromptConfig(
            condense_q_system="Rephrase the question based on history.",
            qa_system="Answer based on context: {context}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return PromptConfig(**data)
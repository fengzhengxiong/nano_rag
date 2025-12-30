#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：helpers.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:01 
'''

import importlib
from typing import Type, Tuple
from pathlib import Path
from functools import lru_cache

from ..core.exceptions import ConfigurationError

# 使用一个 markers 元组，使其更具通用性
DEFAULT_ROOT_MARKERS = ('.project-root', 'pyproject.toml', '.git')

def find_project_root(markers: Tuple[str, ...] = DEFAULT_ROOT_MARKERS) -> Path:
    """
    通过查找一系列标记文件来智能定位项目根目录。
    """
    current_path = Path.cwd().resolve()
    for parent in [current_path] + list(current_path.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent
    raise FileNotFoundError(
        f"Could not find project root. Traversed up from {current_path} but could not find any of these markers: {markers}"
    )


# 增加缓存，避免重复导入，提升性能
@lru_cache(maxsize=128)
def dynamic_import(class_path: str) -> Type:
    """
    根据完整的类路径字符串动态导入并返回类对象。
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ConfigurationError(
            f"Failed to dynamically import class '{class_path}'. "
            f"Please ensure the path is correct and the required libraries are installed."
        ) from e
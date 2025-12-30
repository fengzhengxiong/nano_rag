#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：loader.py
@Author  ：fengzhengxiong
@Date    ：2025/12/29 10:06 
'''

import yaml
import os
import re
import logging
from functools import lru_cache
from pydantic import ValidationError
from pathlib import Path

from .models import AppConfig, ResolvedConfig
from ..utils.helpers import find_project_root
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)
ENV_VAR_MATCHER = re.compile(r"\$\{([^:}]+):?([^}]+)?\}")


def _substitute_env_vars(raw_config_str: str) -> str:
    """在YAML解析前，替换字符串中的 ${VAR:default} 表达式。"""

    def replace(match):
        var_name, default_value = match.groups()
        value = os.environ.get(var_name, default_value)
        if value is None:
            raise ConfigurationError(f"Environment variable '{var_name}' is not set and no default value was provided.")
        return value

    return ENV_VAR_MATCHER.sub(replace, raw_config_str)


@lru_cache(maxsize=1)
def load_app_config(config_file_name: str = "default_config.yaml") -> AppConfig:
    """从项目根目录加载、替换环境变量、校验并返回原始应用配置。"""
    try:
        project_root = find_project_root()
        # 假设 configs 文件夹在项目根目录
        config_path = project_root / "configs" / config_file_name
    except FileNotFoundError as e:
        raise ConfigurationError("Could not locate project root.", original_exception=e)

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found at: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")
    try:
        with config_path.open('r', encoding='utf-8') as f:
            raw_config_content = f.read()

        substituted_config = _substitute_env_vars(raw_config_content)
        raw_config_dict = yaml.safe_load(substituted_config)

        if not raw_config_dict:
            raise ConfigurationError(f"Configuration file '{config_path}' is empty.")

        return AppConfig(**raw_config_dict)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML file '{config_path}'", original_exception=e)
    except ValidationError as e:
        error_messages = "\n".join([f"  - In '{'.'.join(map(str, err['loc']))}': {err['msg']}" for err in e.errors()])
        raise ConfigurationError(f"Configuration validation failed for '{config_path}':\n{error_messages}")
    except Exception as e:
        raise ConfigurationError("An unexpected error occurred while loading config", original_exception=e)


def resolve_active_configs(raw_config: AppConfig) -> ResolvedConfig:
    """将原始配置对象解析为只包含活动配置的 'ResolvedConfig' 对象。"""
    try:
        persist_base_dir = Path(raw_config.paths.persist_base_dir)
        # 动态创建持久化路径，隔离不同模型的数据
        persist_dir = persist_base_dir / raw_config.embedding.active / raw_config.vector_store.active
        persist_dir.mkdir(parents=True, exist_ok=True)

        active_retrieval = raw_config.retrieval_strategy.profiles[raw_config.retrieval_strategy.active]

        return ResolvedConfig(
            logging=raw_config.logging,
            paths=raw_config.paths,
            data_source=raw_config.data_source.profiles[raw_config.data_source.active],
            text_splitter=raw_config.text_splitter.profiles[raw_config.text_splitter.active],
            embedding=raw_config.embedding.profiles[raw_config.embedding.active],
            vector_store=raw_config.vector_store.profiles[raw_config.vector_store.active],
            llm=raw_config.llm.profiles[raw_config.llm.active],
            cache=raw_config.cache.profiles[raw_config.cache.active],
            retrieval_strategy=active_retrieval,
            resolved_paths={"persist_dir": persist_dir}
        )
    except KeyError as e:
        raise ConfigurationError(
            f"Configuration resolution failed. An 'active' profile name '{e}' points to a non-existent profile.")
    except Exception as e:
        raise ConfigurationError(f"An unexpected error occurred during config resolution", original_exception=e)


@lru_cache(maxsize=1)
def get_resolved_config(config_file_name: str = "default_config.yaml") -> ResolvedConfig:
    """
    加载、校验并解析活动的应用程序配置。这是获取配置的唯一推荐入口点。
    """
    raw_config = load_app_config(config_file_name)
    resolved_config = resolve_active_configs(raw_config)
    logger.info("Configuration loaded and resolved successfully.")
    return resolved_config
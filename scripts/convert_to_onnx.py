#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：nano_rag 
@File    ：convert_to_onnx.py
@Author  ：fengzhengxiong
@Date    ：2025/12/30 09:45 
'''

import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer


def convert_reranker():
    # 1. 路径设置
    project_root = Path(__file__).parent.parent
    input_model_path = project_root / "models/bge-reranker-base"
    output_model_path = project_root / "models/bge-reranker-base-onnx"

    print(f"🔄 正在加载 PyTorch 模型: {input_model_path}")

    if not input_model_path.exists():
        print(f"❌ 错误：找不到源模型 {input_model_path}")
        return

    # 2. 导出 ONNX (Export)
    # 这一步会把 PyTorch 权重图转换为 ONNX 计算图
    print("⏳ 正在导出为 ONNX 格式 (这可能需要几分钟)...")
    model = ORTModelForSequenceClassification.from_pretrained(
        input_model_path,
        export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(input_model_path)

    # 3. 量化 (Quantization) -> INT8
    # 针对 CPU (AVX512/AVX2) 进行动态量化
    print("📉 正在进行 INT8 量化...")
    quantizer = ORTQuantizer.from_pretrained(model)
    qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=True)

    quantizer.quantize(
        save_dir=output_model_path,
        quantization_config=qconfig,
    )

    # 保存 Tokenizer (推理时还需要它)
    tokenizer.save_pretrained(output_model_path)

    print(f"✅ 转换完成！量化模型已保存至: {output_model_path}")
    print("👉 文件名: model_quantized.onnx (这是我们要加载的文件)")


if __name__ == "__main__":
    convert_reranker()
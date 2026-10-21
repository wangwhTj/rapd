import torch

torch.set_float32_matmul_precision('medium')

import logging
logging.getLogger("rapd").setLevel(logging.INFO)

import os
os.environ["NCCL_DEBUG"] = "WARN"
# 部分环境（云/容器）下 NVLS 会导致 NCCL init 报 Cuda failure 1 'invalid argument'，关闭后走普通路径
os.environ["NCCL_NVLS_ENABLE"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


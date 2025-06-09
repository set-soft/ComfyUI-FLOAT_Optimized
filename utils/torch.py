# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
import contextlib  # For context manager
import logging
import torch
from .misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.torch")


def get_torch_device_options():
    options = ["cpu"]
    if torch.cuda.is_available():
        options.append("cuda")  # Default CUDA device
        for i in range(torch.cuda.device_count()):
            options.append(f"cuda:{i}")  # Specific CUDA devices
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        options.append("mps")
    return options


# ##################################################################################
# # Helper for cuDNN Benchmark
# ##################################################################################

@contextlib.contextmanager
def manage_cudnn_benchmark(opt):
    """ Context manager to temporarily set and restore cudnn.benchmark state. """
    original_cudnn_benchmark_state = None
    is_cuda_device = opt.rank.type == 'cuda'

    if is_cuda_device and hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
        original_cudnn_benchmark_state = torch.backends.cudnn.benchmark
        if torch.backends.cudnn.benchmark != opt.cudnn_benchmark_enabled:
            torch.backends.cudnn.benchmark = opt.cudnn_benchmark_enabled
            logger.debug(f"Temporarily set cuDNN benchmark to {torch.backends.cudnn.benchmark}")
    try:
        yield
    finally:
        if original_cudnn_benchmark_state is not None:
            torch.backends.cudnn.benchmark = original_cudnn_benchmark_state
            logger.debug(f"Restored cuDNN benchmark to {torch.backends.cudnn.benchmark}")

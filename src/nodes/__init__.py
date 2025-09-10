# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
from seconohe.logger import initialize_logger


__version__ = "1.1.1"
__copyright__ = "Copyright © 2025 Salvador E. Tropea / Instituto Nacional de Tecnología Industrial"
__license__ = "License CC BY-NC-SA 4.0"
__author__ = "Salvador E. Tropea"
NODES_NAME = "FLOAT_Optimized"
EMOTIONS = ['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# List of fixed-step solvers you from torchdiffeq
TORCHDIFFEQ_FIXED_STEP_SOLVERS = [
    "euler",
    "midpoint",
    "rk4",        # Classical Runge-Kutta 4th order
    "heun2",      # Heun's method (RK2)
    "heun3",      # Heun's method (RK3)
    # "explicit_adams", # Fixed-step Adams-Bashforth (multi-step, might need more params like order)
    # "implicit_adams", # Fixed-step Adams-Moulton (multi-step, implicit)
]
FLOAT_URL = "https://huggingface.co/set-soft/float/resolve/main/FLOAT.safetensors?download=true"
FLOAT_UNIFIED_MODEL = "FLOAT.safetensors"

main_logger = initialize_logger(NODES_NAME)

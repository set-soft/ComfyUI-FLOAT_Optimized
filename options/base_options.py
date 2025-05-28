# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
from dataclasses import dataclass


@dataclass
class BaseOptions:
    """
    Base options for the model configuration.
    """
    pretrained_dir: str = "./checkpoints"
    seed: int = 15
    fix_noise_seed: bool = True
    input_size: int = 512
    input_nc: int = 3
    fps: float = 25.0
    sampling_rate: int = 16000
    audio_marcing: int = 2
    wav2vec_sec: float = 2.0
    wav2vec_model_path: str = "./checkpoints/wav2vec2-base-960h"
    audio2emotion_path: str = "./checkpoints/wav2vec-english-speech-emotion-recognition"
    attention_window: int = 2
    only_last_features: bool = False
    average_emotion: bool = False
    audio_dropout_prob: float = 0.1
    ref_dropout_prob: float = 0.1
    emotion_dropout_prob: float = 0.1
    style_dim: int = 512
    dim_a: int = 512
    dim_w: int = 512
    dim_h: int = 1024
    dim_m: int = 20
    dim_e: int = 7
    fmt_depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    no_learned_pe: bool = False
    num_prev_frames: int = 10
    max_grad_norm: float = 1.0
    ode_atol: float = 1e-5
    ode_rtol: float = 1e-5
    nfe: int = 10  # Number of function evaluations
    torchdiffeq_ode_method: str = "euler"
    a_cfg_scale: float = 2.0
    e_cfg_scale: float = 1.0
    r_cfg_scale: float = 1.0
    n_diff_steps: int = 500
    diff_schedule: str = "cosine"
    diffusion_mode: str = "sample"

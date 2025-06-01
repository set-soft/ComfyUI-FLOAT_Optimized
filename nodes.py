# -*- coding: utf-8 -*-
# Copyright (c) 2025 Yuvraj Seegolam
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
import logging
import numpy as np
import os
import torch
import folder_paths
import comfy.model_management as mm

from .generate import InferenceAgent, img_tensor_2_np_array, process_img
from .options.base_options import BaseOptions

# ######################
# Logger setup
# ######################
# 1. Determine the ComfyUI global log level (influenced by --verbose)
float_logger = logging.getLogger("ComfyUI.FLOAT_Nodes")
comfy_root_logger = logging.getLogger('comfy')
effective_comfy_level = logging.getLogger().getEffectiveLevel()
# 2. Check your custom environment variable for more verbosity
try:
    float_nodes_debug_env = int(os.environ.get("FLOAT_NODES_DEBUG", "0"))
except ValueError:
    float_nodes_debug_env = 0
# 3. Set node's logger level
if float_nodes_debug_env:
    float_logger.setLevel(logging.DEBUG - (float_nodes_debug_env - 1))
    final_level_str = f"DEBUG (due to FLOAT_NODES_DEBUG={float_nodes_debug_env})"
else:
    float_logger.setLevel(effective_comfy_level)
    final_level_str = logging.getLevelName(effective_comfy_level) + " (matching ComfyUI global)"
_initial_setup_logger = logging.getLogger(__name__ + ".setup")  # A temporary logger for this message
_initial_setup_logger.info(f"FLOAT_Nodes logger level set to: {final_level_str}")

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


def get_torch_device_options():
    options = ["cpu"]
    if torch.cuda.is_available():
        options.append("cuda")  # Default CUDA device
        for i in range(torch.cuda.device_count()):
            options.append(f"cuda:{i}")  # Specific CUDA devices
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        options.append("mps")
    return options


class LoadFloatModels:
    @classmethod
    def INPUT_TYPES(s):
        device_options = get_torch_device_options()
        default_device = "cuda" if "cuda" in device_options else "cpu"

        float_models_path = os.path.join(folder_paths.models_dir, "float")
        os.makedirs(float_models_path, exist_ok=True)

        # Look for the combined safetensors file
        available_model_files = sorted([f for f in os.listdir(float_models_path) if f.lower().endswith(".safetensors") or
                                        f.lower().endswith(".pth")])
        if not available_model_files:  # Provide a default if none found
            available_model_files = [BaseOptions.ckpt_filename]  # Default from BaseOptions

        return {
            "required": {
                "model": (available_model_files, {"default": BaseOptions.ckpt_filename}),
                "target_device": (device_options, {"default": default_device}),
                "cudnn_benchmark": ("BOOLEAN", {"default": False}, ),
            },
            "optional": {
                "advanced_float_options": ("ADV_FLOAT_DICT",)
            }
        }

    RETURN_TYPES = ("FLOAT_PIPE",)
    RETURN_NAMES = ("float_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Models are auto-downloaded to /ComfyUI/models/float"

    def loadmodel(self, model, target_device, cudnn_benchmark, advanced_float_options=None):
        # Get the root path of this custom node package to locate bundled configs
        node_root_path = os.path.dirname(os.path.abspath(__file__))
        float_logger.debug(f"Node root path for bundled configs: {node_root_path}")

        # Path to the selected combined model file
        ckpt_full_path = os.path.join(folder_paths.models_dir, "float", model)
        float_logger.info(f"Selected combined model file: {ckpt_full_path}")
        float_models_dir = os.path.join(folder_paths.models_dir, "float")

        if model.lower().endswith(".pth"):
            # ##################################
            # Old model with wav2vec separated
            # ##################################
            audio_models_dir = os.path.join(folder_paths.models_dir, "audio")

            wav2vec2_base_960h_models_dir = os.path.join(float_models_dir, "wav2vec2-base-960h")
            wav2vec_english_speech_emotion_recognition_models_dir = os.path.join(float_models_dir,
                                                                                 "wav2vec-english-speech-emotion-recognition")

            # Allow models/audio/wav2vec2-base-960h
            if not os.path.isdir(wav2vec2_base_960h_models_dir):
                alt_dir = os.path.join(audio_models_dir, "wav2vec2-base-960h")
                if os.path.isdir(alt_dir):
                    wav2vec2_base_960h_models_dir = alt_dir
                    float_logger.debug("Using speech encoder from: "+alt_dir)
            # Allow models/audio/wav2vec-english-speech-emotion-recognition
            if not os.path.isdir(wav2vec_english_speech_emotion_recognition_models_dir):
                alt_dir = os.path.join(audio_models_dir, "wav2vec-english-speech-emotion-recognition")
                if os.path.isdir(alt_dir):
                    wav2vec_english_speech_emotion_recognition_models_dir = alt_dir
                    float_logger.debug("Using emotion decoder from: "+alt_dir)

            if (not os.path.exists(ckpt_full_path) or not os.path.isdir(wav2vec2_base_960h_models_dir) or
               not os.path.isdir(wav2vec_english_speech_emotion_recognition_models_dir)):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="yuvraj108c/float", local_dir=float_models_dir, local_dir_use_symlinks=False)

            unified = False
        else:
            # ##############################
            # Unified model (safetensors)
            # ##############################
            if not os.path.exists(ckpt_full_path):
                float_logger.warning(f"Model file {ckpt_full_path} not found. Trying to download it ...")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="set-soft/float", local_dir=float_models_dir)
            unified = True

        # use custom dictionary instead of original parser for arguments
        opt = BaseOptions

        if advanced_float_options is not None and isinstance(advanced_float_options, dict):
            for key, value in advanced_float_options.items():
                if hasattr(opt, key):
                    setattr(opt, key, value)

        opt.rank = torch.device(target_device)
        float_logger.debug(f"Instantiating InferenceAgent for device {opt.rank}")
        opt.cudnn_benchmark = cudnn_benchmark
        opt.ckpt_path = ckpt_full_path
        if not unified:
            opt.pretrained_dir = float_models_dir
            opt.wav2vec_model_path = wav2vec2_base_960h_models_dir
            opt.audio2emotion_path = wav2vec_english_speech_emotion_recognition_models_dir
            float_logger.debug(f"- Using {ckpt_full_path}, {wav2vec2_base_960h_models_dir} and "
                               f"{wav2vec_english_speech_emotion_recognition_models_dir}")
        else:
            float_logger.debug(f"- Using combined model {opt.ckpt_path}")

        agent = InferenceAgent(opt, node_root_path=node_root_path if unified else None)

        float_logger.debug("FLOAT model pipe loaded successfully.")

        return (agent,)


class FloatProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "ref_audio": ("AUDIO",),
                "float_pipe": ("FLOAT_PIPE",),
                "a_cfg_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "step": 0.1}),
                "e_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 25, "step": 1}),
                "emotion": (['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], {"default": "none"}),
                "face_align": ("BOOLEAN", {"default": True}, ),
                "seed": ("INT", {"default": 62064758300528, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "floatprocess"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Float Processing"

    def floatprocess(self, ref_image, ref_audio, float_pipe, a_cfg_scale, e_cfg_scale, fps, emotion, face_align, seed):

        original_cudnn_benchmark_state = None
        is_cuda_device = float_pipe.opt.rank.type == 'cuda'

        try:
            if is_cuda_device and hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
                # Store original state and set new state
                original_cudnn_benchmark_state = torch.backends.cudnn.benchmark
                torch.backends.cudnn.benchmark = float_pipe.opt.cudnn_benchmark_enabled
                float_logger.debug(f"FloatProcess: Temporarily set cuDNN benchmark to {torch.backends.cudnn.benchmark}"
                                   f" (was {original_cudnn_benchmark_state})")
            else:
                float_logger.debug("FloatProcess: Not a CUDA device or cuDNN not available, benchmark setting skipped.")

            float_pipe.G.to(float_pipe.rank)

            float_pipe.opt.fps = fps
            images_bhwc = float_pipe.run_inference(None, ref_image, ref_audio, a_cfg_scale=a_cfg_scale,
                                                   r_cfg_scale=float_pipe.opt.r_cfg_scale, e_cfg_scale=e_cfg_scale,
                                                   emo=None if emotion == "none" else emotion,
                                                   no_crop=not face_align, seed=seed)
            float_pipe.G.to(mm.unet_offload_device())

            return (images_bhwc,)
        finally:
            # Restore original cuDNN benchmark state
            if original_cudnn_benchmark_state is not None:
                torch.backends.cudnn.benchmark = original_cudnn_benchmark_state
                float_logger.debug(f"FloatProcess: Restored cuDNN benchmark to {torch.backends.cudnn.benchmark}")


class FloatImageFaceAlign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_margin": ("FLOAT", {"default": 1.6, "min": 1.2, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Crops the area containing a face and resizes for FLOAT"

    def process(self, image, face_margin):
        np_array_image = img_tensor_2_np_array(image)
        crop_image_np = process_img(np_array_image, BaseOptions.input_size, face_margin)

        return (torch.from_numpy(crop_image_np.astype(np.float32) / 255.0).unsqueeze(0),)


class FloatAdvancedParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "r_cfg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "step": 0.1
                }),
                "attention_window": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,  # Practical maximum
                    "step": 1,
                    "display": "number"
                }),
                "audio_dropout_prob": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "ref_dropout_prob": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "emotion_dropout_prob": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "ode_atol": ("FLOAT", {  # Represent as float, can use scientific notation in input
                    "default": 1e-5,
                    "min": 1e-9,    # Practical minimum for tolerance
                    "max": 1e-1,    # Practical maximum for tolerance
                    "step": 1e-6,   # Adjust step for fine control
                    "display": "number",
                    "precision": 9  # Number of decimal places for display if needed
                }),
                "ode_rtol": ("FLOAT", {
                    "default": 1e-5,
                    "min": 1e-9,
                    "max": 1e-1,
                    "step": 1e-6,
                    "display": "number",
                    "precision": 9
                }),
                "nfe": ("INT", {  # Number of Function Evaluations
                    "default": 10,
                    "min": 1,
                    "max": 1000,  # Practical maximum
                    "step": 1,
                    "display": "number"
                }),
                "torchdiffeq_ode_method": (TORCHDIFFEQ_FIXED_STEP_SOLVERS, {
                    "default": "euler"
                }),
                "face_margin": ("FLOAT", {
                    "default": 1.6,
                    "min": 1.2,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("ADV_FLOAT_DICT",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "get_options"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Float Advanced Options"

    def get_options(self, r_cfg_scale, attention_window, audio_dropout_prob, ref_dropout_prob, emotion_dropout_prob,
                    ode_atol, ode_rtol, nfe, torchdiffeq_ode_method, face_margin):

        options_dict = {
            "r_cfg_scale": r_cfg_scale,
            "attention_window": attention_window,
            "audio_dropout_prob": audio_dropout_prob,
            "ref_dropout_prob": ref_dropout_prob,
            "emotion_dropout_prob": emotion_dropout_prob,
            "ode_atol": ode_atol,
            "ode_rtol": ode_rtol,
            "nfe": nfe,
            "torchdiffeq_ode_method": torchdiffeq_ode_method,
            "face_margin": face_margin,
        }

        return (options_dict,)

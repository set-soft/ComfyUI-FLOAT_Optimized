# -*- coding: utf-8 -*-
# Copyright (c) 2025 Yuvraj Seegolam
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
import numpy as np
import os
import torch
import folder_paths
import comfy.model_management as mm

from .options.base_options import BaseOptions
from .utils.image import img_tensor_2_np_array, process_img
from .utils.logger import float_logger
from .generate import InferenceAgent

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
RGBA_CONVERSION_STRATEGIES = [
    "blend_with_color",
    "discard_alpha",
    "replace_with_color"
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
    UNIQUE_NAME = "LoadFloatModelsOpt"
    DISPLAY_NAME = "Load FLOAT Models (Opt)"

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

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "ref_audio", "fps")
    FUNCTION = "floatprocess"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Float Processing"
    UNIQUE_NAME = "FloatProcessOpt"
    DISPLAY_NAME = "FLOAT Process (Opt)"

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

            # Determine batch sizes
            image_batch_size = ref_image.shape[0]
            audio_waveform = ref_audio['waveform']
            audio_sample_rate = ref_audio['sample_rate']
            audio_batch_size = audio_waveform.shape[0]
            target_batch_size = max(image_batch_size, audio_batch_size)
            all_generated_images = []
            used_audio_waveforms = []

            # Loop for all image/audio pairs
            for i in range(target_batch_size):
                float_logger.debug(f"Processing batch item {i+1}/{target_batch_size}")

                # Select current image slice (B=1, H, W, C)
                current_image_idx = min(i, image_batch_size - 1)
                current_ref_image_slice = ref_image[current_image_idx:current_image_idx+1]
                current_ref_image_slice = current_ref_image_slice.to(float_pipe.rank)

                # Select current audio slice {'waveform': (1, C_aud, N_aud), 'sample_rate': int}
                current_audio_idx = min(i, audio_batch_size - 1)
                current_ref_audio_slice_waveform = audio_waveform[current_audio_idx:current_audio_idx+1]
                current_ref_audio_slice_waveform = current_ref_audio_slice_waveform.to(float_pipe.rank)
                current_ref_audio_dict = {'waveform': current_ref_audio_slice_waveform, 'sample_rate': audio_sample_rate}

                float_logger.debug(current_ref_image_slice.shape)
                float_logger.debug(current_ref_audio_slice_waveform.shape)

                images_thwc = float_pipe.run_inference(None, current_ref_image_slice, current_ref_audio_dict,
                                                       a_cfg_scale=a_cfg_scale, r_cfg_scale=float_pipe.opt.r_cfg_scale,
                                                       e_cfg_scale=e_cfg_scale, emo=None if emotion == "none" else emotion,
                                                       no_crop=not face_align, seed=seed + i)
                all_generated_images.append(images_thwc.cpu())
                used_audio_waveforms.append(current_ref_audio_slice_waveform.cpu())

            if target_batch_size == 1:
                output_audio_dict = ref_audio
            else:
                squeezed_audio_segments = [wf.squeeze(0) for wf in used_audio_waveforms]  # List of (C, N_seg)
                concatenated_channels_first = torch.cat(squeezed_audio_segments, dim=1)   # (C, Total_N)
                final_concatenated_audio_wf = concatenated_channels_first.unsqueeze(0)    # (1, C, Total_N)
                final_concatenated_audio_wf = final_concatenated_audio_wf.to(audio_waveform.device)  # Move to original device
                output_audio_dict = {'waveform': final_concatenated_audio_wf, 'sample_rate': audio_sample_rate}

            return (torch.cat(all_generated_images, dim=0), output_audio_dict, fps,)
        finally:
            # Restore original cuDNN benchmark state
            if original_cudnn_benchmark_state is not None:
                torch.backends.cudnn.benchmark = original_cudnn_benchmark_state
                float_logger.debug(f"FloatProcess: Restored cuDNN benchmark to {torch.backends.cudnn.benchmark}")

            float_pipe.G.to(mm.unet_offload_device())


class FloatImageFaceAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_margin": ("FLOAT", {"default": 1.6, "min": 1.2, "max": 2.0, "step": 0.1}),
                "rgba_conversion": (RGBA_CONVERSION_STRATEGIES, {"default": "blend_with_color"}),
                "bkg_color_hex": ("STRING", {
                    "default": "#000000",  # Black
                    # Only used for "blend_with_color" and "replace_with_color"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Crops Crops to face, resizes, and handles RGBA to RGB conversion."
    UNIQUE_NAME = "FloatImageFaceAlign"
    DISPLAY_NAME = "Face Align for FLOAT"

    def process(self, image, face_margin, rgba_conversion, bkg_color_hex):
        # Defensive: ensure the tensor shape is as documented
        # Should be redundant
        if image.ndim == 3:
            float_logger.warning("Missing batch size")
            image = image.unsqueeze(0)
        if image.ndim != 4:
            msg = f"Image shape is incorrect {image.shape}"
            float_logger.error(msg)
            raise ValueError(msg)

        batch_size = image.shape[0]
        sz = BaseOptions.input_size
        ret = torch.empty((batch_size, sz, sz, 3), dtype=torch.float32, device=image.device)
        for b in range(batch_size):
            np_array_image = img_tensor_2_np_array(image[b], rgba_conversion, bkg_color_hex)
            crop_image_np = process_img(np_array_image, sz, face_margin)
            ret[b] = torch.from_numpy(crop_image_np.astype(np.float32) / 255.0).to(image.device)

        return (ret,)


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
                "rgba_conversion": (RGBA_CONVERSION_STRATEGIES, {"default": "blend_with_color"}),
                "bkg_color_hex": ("STRING", {
                    "default": "#000000",  # Black
                    # Only used for "blend_with_color" and "replace_with_color"
                }),
            }
        }

    RETURN_TYPES = ("ADV_FLOAT_DICT",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "get_options"
    CATEGORY = "FLOAT"
    DESCRIPTION = "FLOAT Advanced Options"
    UNIQUE_NAME = "FloatAdvancedParameters"
    DISPLAY_NAME = "FLOAT Advanced Options"

    def get_options(self, r_cfg_scale, attention_window, audio_dropout_prob, ref_dropout_prob, emotion_dropout_prob,
                    ode_atol, ode_rtol, nfe, torchdiffeq_ode_method, face_margin, rgba_conversion, bkg_color_hex):

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
            "rgba_conversion": rgba_conversion,
            "bkg_color_hex": bkg_color_hex,
        }

        return (options_dict,)

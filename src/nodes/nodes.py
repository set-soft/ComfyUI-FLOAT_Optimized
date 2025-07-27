# -*- coding: utf-8 -*-
# Copyright (c) 2025 Yuvraj Seegolam
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
import os
from seconohe.downloader import download_file
from seconohe.torch import get_torch_device_options, model_to_target, get_canonical_device
import torch
# ComfyUI
import comfy.model_management as mm
import folder_paths

from .options.base_options import BaseOptions
# We are the main source, so we use the main_logger
from . import main_logger, FLOAT_URL, FLOAT_UNIFIED_MODEL
from .generate import InferenceAgent

EMOTIONS = ['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


class LoadFloatModels:
    @classmethod
    def INPUT_TYPES(s):
        device_options, default_device = get_torch_device_options()

        float_models_path = os.path.join(folder_paths.models_dir, "float")
        os.makedirs(float_models_path, exist_ok=True)

        # Look for the combined safetensors file
        available_model_files = sorted([f for f in os.listdir(float_models_path) if f.lower().endswith(".safetensors") or
                                        f.lower().endswith(".pth")])
        if not available_model_files:  # Provide a default if none found
            available_model_files = [BaseOptions.ckpt_filename]  # Default from BaseOptions

        return {
            "required": {
                "model": (available_model_files, {"default": FLOAT_UNIFIED_MODEL}),
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
        main_logger.debug(f"Node root path for bundled configs: {node_root_path}")

        # Path to the selected combined model file
        ckpt_full_path = os.path.join(folder_paths.models_dir, "float", model)
        main_logger.info(f"Selected combined model file: {ckpt_full_path}")
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
                    main_logger.debug("Using speech encoder from: "+alt_dir)
            # Allow models/audio/wav2vec-english-speech-emotion-recognition
            if not os.path.isdir(wav2vec_english_speech_emotion_recognition_models_dir):
                alt_dir = os.path.join(audio_models_dir, "wav2vec-english-speech-emotion-recognition")
                if os.path.isdir(alt_dir):
                    wav2vec_english_speech_emotion_recognition_models_dir = alt_dir
                    main_logger.debug("Using emotion decoder from: "+alt_dir)

            if (not os.path.exists(ckpt_full_path) or not os.path.isdir(wav2vec2_base_960h_models_dir) or
               not os.path.isdir(wav2vec_english_speech_emotion_recognition_models_dir)):
                # One or both audio processors is missing, download the whole pack
                # It can only happen if the user has "float.pth" already downloaded and selects it
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="yuvraj108c/float", local_dir=float_models_dir, local_dir_use_symlinks=False)

            unified = False
        else:
            # ##############################
            # Unified model (safetensors)
            # ##############################
            if not os.path.exists(ckpt_full_path):
                main_logger.warning(f"Model file {ckpt_full_path} not found. Trying to download it...")
                try:
                    download_file(main_logger, url=FLOAT_URL, save_dir=float_models_dir, file_name=FLOAT_UNIFIED_MODEL)
                except Exception:
                    # The downloader already logs, but here we have a final message
                    main_logger.error(f"Failed to download {model}. Please download it manually.")
                    raise  # Re-raise to show the error in the UI

            unified = True

        # Use custom dictionary instead of original parser for arguments
        opt_instance = BaseOptions()  # Create a new instance

        # Apply advanced_float_options if provided
        if advanced_float_options is not None and isinstance(advanced_float_options, dict):
            main_logger.debug(f"Applying advanced_float_options: {advanced_float_options}")
            for key, value in advanced_float_options.items():
                if hasattr(opt_instance, key):
                    setattr(opt_instance, key, value)
                else:
                    main_logger.warning(f"opt_instance has no attribute '{key}' from advanced_float_options.")

        # Set core options
        opt_instance.rank = get_canonical_device(target_device)
        opt_instance.cudnn_benchmark = cudnn_benchmark
        opt_instance.ckpt_path = ckpt_full_path
        if not unified:
            opt_instance.pretrained_dir = float_models_dir
            opt_instance.wav2vec_model_path = wav2vec2_base_960h_models_dir
            opt_instance.audio2emotion_path = wav2vec_english_speech_emotion_recognition_models_dir
            main_logger.debug(f"- Using {ckpt_full_path}, {wav2vec2_base_960h_models_dir} and "
                              f"{wav2vec_english_speech_emotion_recognition_models_dir}")
        else:
            main_logger.debug(f"- Using combined model {opt_instance.ckpt_path}")

        main_logger.debug(f"Instantiating InferenceAgent for device {opt_instance.rank}")
        agent = InferenceAgent(opt_instance, node_root_path=node_root_path if unified else None)
        agent.G.to(mm.unet_offload_device())  # Offload after loading

        main_logger.debug("FLOAT model pipe loaded successfully.")

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
                "emotion": (EMOTIONS, {"default": "none"}),
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

        float_pipe.G.target_device = float_pipe.rank
        float_pipe.G.cudnn_benchmark_setting = float_pipe.opt.cudnn_benchmark_enabled
        with model_to_target(main_logger, float_pipe.G):
            # original_fps = float_pipe.opt.fps
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
                main_logger.debug(f"Processing batch item {i+1}/{target_batch_size}")

                # Select current image slice (B=1, H, W, C)
                current_image_idx = min(i, image_batch_size - 1)
                current_ref_image_slice = ref_image[current_image_idx:current_image_idx+1]
                current_ref_image_slice = current_ref_image_slice.to(float_pipe.rank)

                # Select current audio slice {'waveform': (1, C_aud, N_aud), 'sample_rate': int}
                current_audio_idx = min(i, audio_batch_size - 1)
                current_ref_audio_slice_waveform = audio_waveform[current_audio_idx:current_audio_idx+1]
                current_ref_audio_slice_waveform = current_ref_audio_slice_waveform.to(float_pipe.rank)
                current_ref_audio_dict = {'waveform': current_ref_audio_slice_waveform, 'sample_rate': audio_sample_rate}

                main_logger.debug(current_ref_image_slice.shape)
                main_logger.debug(current_ref_audio_slice_waveform.shape)

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

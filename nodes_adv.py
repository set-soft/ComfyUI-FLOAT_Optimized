# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
# ###############################################################################################################
# Advanced nodes
# ###############################################################################################################
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchdiffeq import odeint
# from tqdm import tqdm
from typing import List, Dict  # , NewType
# ComfyUI
import comfy.model_management as mm
import comfy.utils

from .options.base_options import BaseOptions
from .utils.image import img_tensor_2_np_array, process_img
from .utils.torch import manage_cudnn_benchmark
from .utils.misc import EMOTIONS, NODES_NAME
from .generate import InferenceAgent

BASE_CATEGORY = "FLOAT/Advanced"
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
logger = logging.getLogger(f"{NODES_NAME}.nodes_vadv")


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
    CATEGORY = BASE_CATEGORY
    DESCRIPTION = "Crops Crops to face, resizes, and handles RGBA to RGB conversion."
    UNIQUE_NAME = "FloatImageFaceAlign"
    DISPLAY_NAME = "Face Align for FLOAT"

    def process(self, image, face_margin, rgba_conversion, bkg_color_hex):
        # Defensive: ensure the tensor shape is as documented
        # Should be redundant
        if image.ndim == 3:
            logger.warning("Missing batch size")
            image = image.unsqueeze(0)
        if image.ndim != 4:
            msg = f"Image shape is incorrect {image.shape}"
            logger.error(msg)
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
    CATEGORY = BASE_CATEGORY
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


class FloatEncodeImageToLatents:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE",),  # ComfyUI standard image batch
                "float_pipe": ("FLOAT_PIPE",),
            }
        }

    # Define a unique string for the custom feats dictionary type
    RETURN_TYPES = ("TORCH_TENSOR", "FLOAT_FEATS_DICT", "TORCH_TENSOR", "FLOAT_PIPE")
    RETURN_NAMES = ("s_r_latent", "s_r_feats_dict", "r_s_lambda_latent", "float_pipe")
    FUNCTION = "encode_image_batch"
    CATEGORY = BASE_CATEGORY
    DESCRIPTION = "Encodes a batch of reference images into FLOAT latents (s_r, r_s_lambda, s_r_feats)."
    UNIQUE_NAME = "FloatEncodeImageToLatents"
    DISPLAY_NAME = "FLOAT Encode Image to Latents"

    def encode_image_batch(self, ref_image: torch.Tensor, float_pipe: InferenceAgent):
        agent = float_pipe
        opt = agent.opt  # Access options like input_size, input_nc, rank

        # --- Input Validation ---
        if not isinstance(ref_image, torch.Tensor):
            raise TypeError(f"Input 'ref_image' must be a torch.Tensor, got {type(ref_image)}")
        if ref_image.ndim != 4:
            raise ValueError(f"Input 'ref_image' must be a 4D tensor (B, H, W, C), got {ref_image.ndim}D.")

        batch_size, height, width, channels = ref_image.shape

        if height != opt.input_size or width != opt.input_size:
            raise ValueError(f"Input images must be {opt.input_size}x{opt.input_size}. "
                             f"Got {height}x{width}.")
        if channels != opt.input_nc:  # opt.input_nc is typically 3 (RGB)
            raise ValueError(f"Input images must be RGB (channels={opt.input_nc}). "
                             f"Got {channels} channels.")

        local_comfy_pbar = None
        original_agent_pbar_ref = None

        with manage_cudnn_benchmark(opt):
            try:
                agent.G.to(opt.rank)

                # --- Progress Bar Handling for First Run of this specific component ---
                if agent.G.first_run:
                    num_encoder_layers = len(agent.G.motion_autoencoder.enc.net_app.convs)
                    # Progress per layer for the entire batch (PyTorch handles batching within conv layers)
                    local_comfy_pbar = comfy.utils.ProgressBar(num_encoder_layers)
                    if hasattr(agent.G, 'pbar'):
                        original_agent_pbar_ref = agent.G.pbar
                    agent.G.pbar = local_comfy_pbar
                    logger.info("First run: Using progress bar for image encoding.")

                # --- Image Preprocessing (ComfyUI IMAGE to Model Input) ---
                # Input: (B, H, W, C), float32, range [0,1]
                # Expected by model's encoder: (B, C, H, W), float32, range [-1,1]
                # - Permute C to be after B: (B, C, H, W)
                s_for_encoder = ref_image.permute(0, 3, 1, 2).contiguous()
                # - Normalize from [0,1] to [-1,1]
                s_for_encoder = (s_for_encoder * 2.0) - 1.0
                s_for_encoder = s_for_encoder.to(opt.rank)

                # --- Core Encoding Operation ---
                s_r, r_s_lambda, s_r_feats_list_gpu = agent.G.encode_image_into_latent(s_for_encoder)
                # s_r: (B, dim_w)
                # r_s_lambda: (B, dim_m)
                # s_r_feats_list_gpu: List of Tensors, each (B, C_feat, H_feat, W_feat)

                # --- Output Formatting ---
                s_r_latent_cpu = s_r.cpu()
                r_s_lambda_latent_cpu = r_s_lambda.cpu()
                s_r_feats_cpu_list = [feat.cpu() for feat in s_r_feats_list_gpu]
                s_r_feats_dict_cpu = {"value": s_r_feats_cpu_list}

                # This node does not set agent.G.first_run = False.
                # That should be handled by a node that completes the full pipeline
                # or if this component itself is considered fully "warmed up".
                return (s_r_latent_cpu, s_r_feats_dict_cpu, r_s_lambda_latent_cpu, float_pipe)

            finally:
                # --- Restore Pbar State (if changed) ---
                if agent.G.first_run:  # Only if pbar could have been set
                    if hasattr(agent.G, 'pbar') and original_agent_pbar_ref is not None:
                        agent.G.pbar = original_agent_pbar_ref
                    elif hasattr(agent.G, 'pbar') and local_comfy_pbar is agent.G.pbar:
                        agent.G.pbar = None
                agent.G.to(mm.unet_offload_device())


class FloatGetIdentityReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "r_s_lambda_latent": ("TORCH_TENSOR",),
                "float_pipe": ("FLOAT_PIPE",),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR", "FLOAT_PIPE")
    RETURN_NAMES = ("r_s_latent", "float_pipe")
    FUNCTION = "get_identity_reference_batch"
    CATEGORY = BASE_CATEGORY
    DESCRIPTION = "Derives the batched identity reference latent (r_s) from r_s_lambda."
    UNIQUE_NAME = "FloatGetIdentityReference"
    DISPLAY_NAME = "FLOAT Get Identity Reference"

    def get_identity_reference_batch(self, r_s_lambda_latent: torch.Tensor, float_pipe: InferenceAgent):
        agent = float_pipe
        opt = agent.opt

        # --- Input Validation ---
        if not isinstance(r_s_lambda_latent, torch.Tensor):
            raise TypeError(f"Input 'r_s_lambda_latent' must be a torch.Tensor, "
                            f"got {type(r_s_lambda_latent)}")
        # r_s_lambda is (B, dim_m), so 2D
        if r_s_lambda_latent.ndim != 2:
            raise ValueError(f"Input 'r_s_lambda_latent' must be a 2D tensor (Batch, DimM), "
                             f"got {r_s_lambda_latent.ndim}D.")
        if r_s_lambda_latent.shape[1] != opt.dim_m:
            raise ValueError(f"Dimension 1 of 'r_s_lambda_latent' should be opt.dim_m ({opt.dim_m}), "
                             f"got {r_s_lambda_latent.shape[1]}.")

        with manage_cudnn_benchmark(opt):
            try:
                agent.G.motion_autoencoder.dec.to(opt.rank)
                r_s_lambda_dev = r_s_lambda_latent.to(opt.rank)

                # --- Core Operation ---
                # agent.G.motion_autoencoder.dec.direction() should handle a batch input for r_s_lambda.
                # If r_s_lambda_dev is (B, dim_m), then r_s_latent should be (B, style_dim)
                # where style_dim is derived from the weights of the Direction layer (512 in original code).
                r_s_latent_batch_gpu = agent.G.motion_autoencoder.dec.direction(r_s_lambda_dev)

                # --- Output Formatting ---
                r_s_latent_batch_cpu = r_s_latent_batch_gpu.cpu()

                return (r_s_latent_batch_cpu, float_pipe)

            finally:
                agent.G.motion_autoencoder.dec.to(mm.unet_offload_device())


class FloatEncodeAudioToLatentWA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_pipe": ("FLOAT_PIPE",),
                "audio": ("AUDIO",),
                # This FPS determines the video length, not the internal chunking
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR", "INT", "TORCH_TENSOR", "FLOAT_PIPE")
    RETURN_NAMES = ("wa_latent", "audio_num_frames", "preprocessed_audio", "float_pipe")
    FUNCTION = "encode_audio_to_wa"
    CATEGORY = BASE_CATEGORY
    DESCRIPTION = "Resamples, preprocesses a batch of audio (uniform length), and encodes it into WA."
    UNIQUE_NAME = "FloatEncodeAudioToLatentWA"
    DISPLAY_NAME = "FLOAT Encode Audio to latent wa"

    def encode_audio_to_wa(self, float_pipe: InferenceAgent, audio: Dict, fps: float):
        agent = float_pipe
        opt = agent.opt

        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise TypeError("Input 'audio' must be a ComfyUI AUDIO dictionary.")
        if not isinstance(audio["waveform"], torch.Tensor):
            raise TypeError("audio['waveform'] must be a torch.Tensor.")

        input_waveform_batch = audio["waveform"]  # Expected Shape: (B, NumChannels, NumSamples)
        input_sample_rate = audio["sample_rate"]

        if input_waveform_batch.ndim == 2:  # (Channels, Samples) -> unsqueeze to (1, C, S)
            input_waveform_batch = input_waveform_batch.unsqueeze(0)
            logger.debug("Input audio waveform was 2D, unsqueezed to 3D for batch processing.")
        elif input_waveform_batch.ndim != 3:
            raise ValueError(f"audio['waveform'] must be 2D or 3D, got {input_waveform_batch.ndim}D.")

        batch_size = input_waveform_batch.shape[0]
        num_samples_input = input_waveform_batch.shape[2]  # Assuming uniform S

        preprocessed_audio_list: List[torch.Tensor] = []

        logger.info(f"Preprocessing batch of {batch_size} audio clip(s) each with {num_samples_input} samples...")
        # Since default_aud_loader only processes the first item of a batch, we loop.
        for i in range(batch_size):
            current_waveform_for_loader = input_waveform_batch[i].unsqueeze(0)  # Shape: (1, NumChannels, NumSamples)

            single_audio_item_dict = {
                "waveform": current_waveform_for_loader,
                "sample_rate": input_sample_rate
            }

            # Output shape: (NumSamplesAfterPrep)
            processed_single_audio_cpu = agent.data_processor.default_aud_loader(single_audio_item_dict)
            preprocessed_audio_list.append(processed_single_audio_cpu)

        # Stack into a batch: (BatchSize, NumSamplesAfterPrep)
        try:
            preprocessed_audio_batched_cpu = torch.stack(preprocessed_audio_list, dim=0)
        except RuntimeError as e:
            if "stack expects each tensor to be equal size" in str(e):
                raise RuntimeError(
                    "Failed to stack preprocessed audio clips due to varying lengths after "
                    "DataProcessor.default_aud_loader. This indicates that either the input audio "
                    "clips, despite having the same raw sample count, resulted in different processed lengths, "
                    "or default_aud_loader is not perfectly deterministic in length output for uniform inputs. "
                    "The input AUDIO batch must result in uniform processed lengths."
                ) from e
            raise e  # Re-raise other runtime errors

        num_samples_after_prep = preprocessed_audio_batched_cpu.shape[1]

        # original_agent_opt_fps = opt.fps
        opt.fps = fps

        with manage_cudnn_benchmark(opt):
            try:
                agent.G.audio_encoder.to(opt.rank)
                audio_on_device = preprocessed_audio_batched_cpu.to(opt.rank)

                # Calculate Number of Frames based on the now uniform num_samples_after_prep
                audio_num_frames = math.ceil(num_samples_after_prep * opt.fps / opt.sampling_rate)
                logger.info(f"Common audio_num_frames for batch: {audio_num_frames} (from {num_samples_after_prep} "
                            f"processed audio samples, {opt.fps} video_fps, {opt.sampling_rate} audio_sr).")

                # Output shape: (BatchSize, audio_num_frames, DimW)
                wa_latent_gpu = agent.G.audio_encoder.inference(audio_on_device, seq_len=audio_num_frames)

                wa_latent_cpu = wa_latent_gpu.cpu()

                return (wa_latent_cpu, audio_num_frames, preprocessed_audio_batched_cpu, float_pipe)

            finally:
                # opt.fps = original_agent_opt_fps
                agent.G.audio_encoder.to(mm.unet_offload_device())


class FloatEncodeEmotionToLatentWE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preprocessed_audio": ("TORCH_TENSOR",),  # Output from FloatEncodeAudioToLatentWA
                "float_pipe": ("FLOAT_PIPE",),
                "emotion": (EMOTIONS, {"default": "none"}),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR", "FLOAT_PIPE")
    RETURN_NAMES = ("we_latent", "float_pipe")
    FUNCTION = "encode_emotion_to_we"
    CATEGORY = BASE_CATEGORY
    DESCRIPTION = "Encodes emotion (from audio or specified) into the WE latent."
    UNIQUE_NAME = "FloatEncodeEmotionToLatentWE"
    DISPLAY_NAME = "FLOAT Encode Emotion to latent we"

    def encode_emotion_to_we(self, preprocessed_audio: torch.Tensor, float_pipe: InferenceAgent, emotion: str):
        agent = float_pipe
        opt = agent.opt

        # --- Input Validation ---
        if not isinstance(preprocessed_audio, torch.Tensor):
            raise TypeError(f"Input 'preprocessed_audio' must be a torch.Tensor, "
                            f"got {type(preprocessed_audio)}")
        # Expected shape: (B, NumSamplesAfterPrep)
        if preprocessed_audio.ndim != 2:
            raise ValueError(f"Input 'preprocessed_audio' must be a 2D tensor (Batch, NumSamples), "
                             f"got {preprocessed_audio.ndim}D with shape {preprocessed_audio.shape}.")

        batch_size = preprocessed_audio.shape[0]

        with manage_cudnn_benchmark(opt):
            try:
                # emotion_encoder contains Wav2Vec2ForSpeechClassification model
                agent.G.emotion_encoder.to(opt.rank)
                audio_on_device = preprocessed_audio.to(opt.rank)

                device_for_one_hot = opt.rank  # Target device for one_hot tensor

                emo_label_lower = str(emotion).lower()
                emo_idx = agent.G.emotion_encoder.label2id.get(emo_label_lower, None)

                we_latent_gpu = None

                if emo_idx is None or emo_label_lower == "none":
                    logger.info("Predicting emotion from audio batch.")
                    # predict_emotion expects (B, NumSamples) and returns (B, NumClasses)
                    # It calls self.wav2vec2_for_emotion.forward(a).logits
                    # and then F.softmax(logits, dim=1)
                    # Wav2Vec2ForSpeechClassification.forward should handle batch input.
                    predicted_scores_batch = agent.G.emotion_encoder.predict_emotion(audio_on_device)  # (B, NumClasses)
                    we_latent_gpu = predicted_scores_batch.unsqueeze(1)  # (B, 1, NumClasses)
                else:
                    logger.info(f"Using specified emotion: {emotion} for batch size {batch_size}.")
                    # Create a one-hot tensor for the specified emotion
                    one_hot_single = F.one_hot(torch.tensor(emo_idx, device=device_for_one_hot),
                                               num_classes=opt.dim_e).float()  # (NumClasses)
                    # Repeat for batch and add sequence dimension
                    we_latent_gpu = one_hot_single.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 1, NumClasses)

                # --- Output Formatting ---
                we_latent_cpu = we_latent_gpu.cpu()

                return (we_latent_cpu, float_pipe)

            finally:
                agent.G.emotion_encoder.to(mm.unet_offload_device())


class FloatSampleMotionSequenceRD:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "r_s_latent": ("TORCH_TENSOR",),
                "wa_latent": ("TORCH_TENSOR",),
                "audio_num_frames": ("INT", {"forceInput": True}),
                "we_latent": ("TORCH_TENSOR",),
                "float_pipe": ("FLOAT_PIPE",),
                "a_cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "e_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 62064758300528, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR", "FLOAT_PIPE")
    RETURN_NAMES = ("r_d_latents", "float_pipe")
    FUNCTION = "sample_rd_sequence"
    CATEGORY = BASE_CATEGORY
    DESCRIPTION = "Samples RD using FMT and ODE, with some ODE params from pipe's options."
    UNIQUE_NAME = "FloatSampleMotionSequenceRD"
    DISPLAY_NAME = "FLOAT Sample Motion Sequence rd"

    def sample_rd_sequence(self, r_s_latent: torch.Tensor, wa_latent: torch.Tensor,
                           audio_num_frames: int, we_latent: torch.Tensor, float_pipe: InferenceAgent,
                           a_cfg_scale: float, e_cfg_scale: float, seed: int):
        agent = float_pipe
        opt = agent.opt

        # --- Input Validation ---
        if not all(isinstance(t, torch.Tensor) for t in [r_s_latent, wa_latent, we_latent]):
            raise TypeError("All latent inputs (r_s, wa, we) must be torch.Tensors.")

        batch_size = wa_latent.shape[0]
        if not (r_s_latent.shape[0] == batch_size and we_latent.shape[0] == batch_size):
            raise ValueError(f"Batch size mismatch: wa_latent has {batch_size}, "
                             f"r_s_latent has {r_s_latent.shape[0]}, "
                             f"we_latent has {we_latent.shape[0]}. All must match.")

        if wa_latent.shape[1] != audio_num_frames:
            logger.warning(f"wa_latent time dimension ({wa_latent.shape[1]}) "
                           f"differs from audio_num_frames ({audio_num_frames}). "
                           f"Using wa_latent.shape[1] for chunking if shorter, "
                           f"or audio_num_frames for trimming if wa_latent is longer.")
            # The loop below uses audio_num_frames for total chunks,
            # and slices/pads wa_latent_dev based on chunk_idx.
            # Final trimming is to audio_num_frames. This should be robust.

        # --- Use ODE parameters from agent.opt ---
        nfe_to_use = opt.nfe
        ode_method_to_use = opt.torchdiffeq_ode_method
        ode_atol_to_use = opt.ode_atol
        ode_rtol_to_use = opt.ode_rtol
        # r_cfg_scale is also from opt
        r_cfg_scale_to_use = opt.r_cfg_scale

        logger.info(f"Using ODE params from pipe: NFE={nfe_to_use}, Method={ode_method_to_use}, "
                    f"ATOL={ode_atol_to_use}, RTOL={ode_rtol_to_use}. R_CFG={r_cfg_scale_to_use}")

        current_odeint_kwargs = {
            'atol': ode_atol_to_use,
            'rtol': ode_rtol_to_use,
            'method': ode_method_to_use
        }
        time_linspace = torch.linspace(0, 1, nfe_to_use, device=opt.rank)

        if opt.fix_noise_seed:
            g = torch.Generator(opt.rank)
            final_seed = opt.seed if seed == -1 else seed
            g.manual_seed(final_seed)
            logger.debug(f"Using fixed seed: {final_seed} for ODE sampling.")
        else:
            g = None
            logger.debug("Not using a seed for ODE sampling.")

        sampled_chunks_list = []
        prev_x_batch = torch.zeros(batch_size, opt.num_prev_frames, opt.dim_w, device=opt.rank)
        prev_wa_batch = torch.zeros(batch_size, opt.num_prev_frames, opt.dim_w, device=opt.rank)

        model_num_prev_frames = agent.G.num_prev_frames
        model_num_frames_for_clip = agent.G.num_frames_for_clip
        total_num_chunks = math.ceil(audio_num_frames / model_num_frames_for_clip)
        comfy_pbar = comfy.utils.ProgressBar(total_num_chunks)
        logger.info(f"Starting ODE sampling for {batch_size} item(s) over {total_num_chunks} chunks.")

        with manage_cudnn_benchmark(opt):
            try:
                agent.G.fmt.to(opt.rank)
                r_s_latent_dev = r_s_latent.to(opt.rank)
                we_latent_dev = we_latent.to(opt.rank)
                wa_latent_dev = wa_latent.to(opt.rank)

                for chunk_idx in range(total_num_chunks):
                    x0_chunk_batch = torch.randn(batch_size, model_num_frames_for_clip, opt.dim_w,
                                                 device=opt.rank, generator=g)

                    start_idx = chunk_idx * model_num_frames_for_clip
                    end_idx = (chunk_idx + 1) * model_num_frames_for_clip
                    wa_chunk_batch = wa_latent_dev[:, start_idx:end_idx, :]

                    current_chunk_len = wa_chunk_batch.shape[1]
                    if current_chunk_len < model_num_frames_for_clip:
                        padding_size = model_num_frames_for_clip - current_chunk_len
                        wa_chunk_batch = F.pad(wa_chunk_batch, (0, 0, 0, padding_size), mode='replicate')

                    def fmt_ode_func_batch(t_scalar, current_x_batch):
                        t_for_fmt = t_scalar.unsqueeze(0)
                        if t_scalar.ndim == 1 and t_scalar.shape[0] == 1:
                            t_for_fmt = t_scalar
                        elif t_scalar.ndim > 0:
                            raise ValueError(f"Unexpected time tensor shape from odeint: {t_scalar.shape}")

                        output_combined_batch = agent.G.fmt.forward_with_cfv(
                            t=t_for_fmt,
                            x=current_x_batch,
                            wa=wa_chunk_batch,
                            wr=r_s_latent_dev,
                            we=we_latent_dev,
                            prev_x=prev_x_batch,
                            prev_wa=prev_wa_batch,
                            a_cfg_scale=a_cfg_scale,
                            r_cfg_scale=r_cfg_scale_to_use,  # Using value from opt
                            e_cfg_scale=e_cfg_scale
                        )
                        return output_combined_batch[:, model_num_prev_frames:, :]

                    trajectory_batch = odeint(fmt_ode_func_batch, x0_chunk_batch, time_linspace, **current_odeint_kwargs)
                    current_sample_output_batch = trajectory_batch[-1]
                    sampled_chunks_list.append(current_sample_output_batch)

                    if current_sample_output_batch.shape[1] >= model_num_prev_frames:
                        prev_x_batch = current_sample_output_batch[:, -model_num_prev_frames:, :]
                    else:
                        prev_x_batch = F.pad(current_sample_output_batch, (0, 0, 0, model_num_prev_frames -
                                             current_sample_output_batch.shape[1]),
                                             mode='replicate')[:, -model_num_prev_frames:, :]

                    if wa_chunk_batch.shape[1] >= model_num_prev_frames:
                        prev_wa_batch = wa_chunk_batch[:, -model_num_prev_frames:, :]
                    else:
                        prev_wa_batch = F.pad(wa_chunk_batch, (0, 0, 0, model_num_prev_frames - wa_chunk_batch.shape[1]),
                                              mode='replicate')[:, -model_num_prev_frames:, :]

                    comfy_pbar.update(1)

                r_d_latents_full_gpu = torch.cat(sampled_chunks_list, dim=1)
                r_d_latents_gpu = r_d_latents_full_gpu[:, :audio_num_frames, :]
                r_d_latents_cpu = r_d_latents_gpu.cpu()

                return (r_d_latents_cpu, float_pipe)

            finally:
                agent.G.fmt.to(mm.unet_offload_device())


class FloatDecodeLatentsToImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s_r_latent": ("TORCH_TENSOR",),          # (B, DimW)
                "s_r_feats_dict": ("FLOAT_FEATS_DICT",),  # Dict: {"value": List[(B, C, H, W), ...]}
                "r_d_latents": ("TORCH_TENSOR",),         # (B, NumFrames, DimW)
                "float_pipe": ("FLOAT_PIPE",),
            }
        }

    RETURN_TYPES = ("IMAGE",  # ComfyUI Standard: (B*T, H, W, C) on CPU
                    "FLOAT", "FLOAT_PIPE")
    RETURN_NAMES = ("images", "fps", "float_pipe")
    FUNCTION = "decode_latents_to_images"
    CATEGORY = BASE_CATEGORY
    DESCRIPTION = "Decodes FLOAT latents (SR, SR_feats, RD) into an image sequence."
    UNIQUE_NAME = "FloatDecodeLatentsToImages"
    DISPLAY_NAME = "FLOAT Decode Latents to Images"

    def decode_latents_to_images(self, s_r_latent: torch.Tensor,
                                 s_r_feats_dict: Dict[str, List[torch.Tensor]],
                                 r_d_latents: torch.Tensor,
                                 float_pipe: InferenceAgent):
        agent = float_pipe
        opt = agent.opt

        # --- Input Validation ---
        if not all(isinstance(t, torch.Tensor) for t in [s_r_latent, r_d_latents]):
            raise TypeError("Inputs s_r_latent and r_d_latents must be torch.Tensors.")
        if not isinstance(s_r_feats_dict, dict) or "value" not in s_r_feats_dict or \
           not isinstance(s_r_feats_dict["value"], list):
            raise TypeError("s_r_feats_dict must be a dict with key 'value' containing a list of tensors.")
        if not all(isinstance(t, torch.Tensor) for t in s_r_feats_dict["value"]):
            raise TypeError("All elements in s_r_feats_dict['value'] must be torch.Tensors.")

        batch_size = s_r_latent.shape[0]
        if not (r_d_latents.shape[0] == batch_size and
                all(feat.shape[0] == batch_size for feat in s_r_feats_dict["value"])):
            s_r_feats_batch_sizes = [feat.shape[0] for feat in s_r_feats_dict["value"]]
            raise ValueError(
                f"Batch size mismatch: s_r_latent {batch_size}, r_d_latents {r_d_latents.shape[0]}, "
                f"s_r_feats batch sizes {s_r_feats_batch_sizes}. All must match."
            )

        num_frames_per_sequence = r_d_latents.shape[1]  # T dimension
        if num_frames_per_sequence == 0:
            logger.warning("r_d_latents has 0 frames. Returning empty image batch.")
            dummy_h, dummy_w, dummy_c = opt.input_size, opt.input_size, opt.input_nc
            # Shape for IMAGE is (N, H, W, C). If T=0, then N = B*0 = 0.
            empty_images = torch.empty((0, dummy_h, dummy_w, dummy_c), dtype=torch.float32, device='cpu')
            return (empty_images, opt.fps, float_pipe)

        original_agent_pbar_ref = None
        # Total frames to show progress for is B * T
        comfy_pbar_decode = comfy.utils.ProgressBar(num_frames_per_sequence * batch_size)

        if hasattr(agent.G, 'pbar'):
            original_agent_pbar_ref = agent.G.pbar
        agent.G.pbar = comfy_pbar_decode

        with manage_cudnn_benchmark(opt):
            try:
                agent.G.motion_autoencoder.dec.to(opt.rank)

                s_r_dev = s_r_latent.to(opt.rank)
                s_r_feats_dev_list = [feat.to(opt.rank) for feat in s_r_feats_dict["value"]]
                r_d_dev = r_d_latents.to(opt.rank)

                decoded_image_sequences_list = []  # To collect (T, H, W, C) tensors

                if batch_size > 1:
                    logger.info(f"Decoding batch of {batch_size} items, each with {num_frames_per_sequence} frames.")

                for b_idx in range(batch_size):
                    s_r_item = s_r_dev[b_idx].unsqueeze(0)
                    s_r_feats_item = [feat[b_idx].unsqueeze(0) for feat in s_r_feats_dev_list]
                    r_d_item = r_d_dev[b_idx].unsqueeze(0)

                    # agent.G.decode_latent_into_processed_images is assumed to take B=1 input
                    # and return (T, H, W, C) for that single batch item.
                    # Its internal progress bar will be updated T times for this item.
                    images_thwc_cpu_item = agent.G.decode_latent_into_processed_images(
                        s_r=s_r_item,
                        s_r_feats=s_r_feats_item,
                        r_d=r_d_item)  # Output: (T, H, W, C)
                    decoded_image_sequences_list.append(images_thwc_cpu_item)

                # Concatenate along dim=0 to get (B*T, H, W, C)
                if not decoded_image_sequences_list:  # Should not happen if batch_size > 0
                    final_images_output_cpu = torch.empty((0, opt.input_size, opt.input_size, opt.input_nc),
                                                          dtype=torch.float32, device='cpu')
                else:
                    final_images_output_cpu = torch.cat(decoded_image_sequences_list, dim=0)

                return (final_images_output_cpu, opt.fps, float_pipe)

            finally:
                if hasattr(agent.G, 'pbar') and original_agent_pbar_ref is not None:
                    agent.G.pbar = original_agent_pbar_ref
                elif hasattr(agent.G, 'pbar') and comfy_pbar_decode is agent.G.pbar:
                    agent.G.pbar = None

                agent.G.motion_autoencoder.dec.to(mm.unet_offload_device())

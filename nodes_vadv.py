# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
import logging
import math
import os
import torch
import torch.nn.functional as F
# from tqdm import tqdm
from typing import List, Dict  # , NewType
# ComfyUI
import comfy.utils
import comfy.model_management as mm
import folder_paths

from .options.base_options import BaseOptions
from .utils.logger import main_logger
from .utils.torch import get_torch_device_options, manage_cudnn_benchmark
from .utils.misc import EMOTIONS, NODES_NAME
from .models.float.encoder import Encoder as FloatEncoderModule
from .models.float.styledecoder import Synthesis as FloatSynthesisModule
from .models.misc import CHANNELS_MAP

logger = logging.getLogger(f"{NODES_NAME}.nodes_vadv")
PROJECTIONS_DIR = "float/audio_projections"
MOTION_AE_DIR = "float/motion_autoencoder"
BASE_CATEGORY = "FLOAT/Very Advanced"
ESPR = "wav2vec-english-speech-emotion-recognition"
SUFFIX = "(VA)"


class LoadWav2VecModel:
    UNIQUE_NAME = "LoadWav2VecModel"  # Static class attribute for registration
    DISPLAY_NAME = "Load Wav2Vec Model (for Audio Encoding)"  # Static class attribute

    @classmethod
    def INPUT_TYPES(cls):
        audio_models_path = os.path.join(folder_paths.models_dir, "audio")
        if not os.path.isdir(audio_models_path):
            # Attempt to create it, but don't fail if it doesn't exist yet,
            # as the list might just show "No models found".
            try:
                os.makedirs(audio_models_path, exist_ok=True)
            except OSError:
                pass  # Oh well, list will be empty or show error.

        model_folders = ["No models found in models/audio/"]  # Default if empty or path issue
        if os.path.isdir(audio_models_path):
            folders = []
            for f_name in os.listdir(audio_models_path):
                full_path = os.path.join(audio_models_path, f_name)
                if os.path.isdir(full_path):
                    # Check for common Hugging Face model files
                    if os.path.exists(os.path.join(full_path, "config.json")) and \
                        (os.path.exists(os.path.join(full_path, "pytorch_model.bin")) or
                         os.path.exists(os.path.join(full_path, "model.safetensors")) or
                         os.path.exists(os.path.join(full_path, "tf_model.h5"))):  # Added safetensors and tf
                        folders.append(f_name)
            if folders:  # If any valid folders were found
                model_folders = sorted(folders)

        device_options = get_torch_device_options()
        default_device = "cuda" if "cuda" in device_options else "cpu"
        default_model = "wav2vec2-base-960h" if "wav2vec2-base-960h" in model_folders else model_folders[0]

        return {
            "required": {
                "model_folder": (model_folders, {"default": default_model}),
                "target_device": (device_options, {"default": default_device}),
            },
            "optional": {
                # This dict would come from FloatAdvancedParameters or be manually created
                "advanced_float_options": ("ADV_FLOAT_DICT",)
            }
        }

    RETURN_TYPES = ("WAV2VEC_PIPE",)  # Pipe: (model, feature_extractor, effective_options)
    RETURN_NAMES = ("wav2vec_pipe",)
    FUNCTION = "load_float_wav2vec_model"
    CATEGORY = BASE_CATEGORY + "/Loaders"

    def load_float_wav2vec_model(self, model_folder: str, target_device: str, advanced_float_options: Dict = None):
        # We import HFWav2Vec2Model only to load its state_dict, not to return it.
        from transformers import Wav2Vec2Model as HFWav2Vec2ModelForLoadingWeights
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import AutoConfig
        from copy import deepcopy
        # Import your custom model class
        from .models.wav2vec2 import Wav2VecModel as FloatWav2VecModel

        if model_folder == "No models found in models/audio/":
            raise FileNotFoundError("No Wav2Vec models found. Place Hugging Face model folders into 'ComfyUI/models/audio/'.")

        model_path = os.path.join(folder_paths.models_dir, "audio", model_folder)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Selected model folder not found: {model_path}")

        main_logger.info(f"Loading HF Wav2Vec weights from: {model_path} into FloatWav2VecModel, target device "
                         f"{target_device}")

        device = torch.device(target_device)

        try:
            # Load configuration
            config = AutoConfig.from_pretrained(model_path)

            # Instantiate YOUR FloatWav2VecModel with this config
            # Your FloatWav2VecModel.__init__(self, config) is used here.
            float_wav2vec_instance = FloatWav2VecModel(config)
            main_logger.debug(f"Instantiated custom FloatWav2VecModel with config from {model_folder}")

            # Load the pre-trained Hugging Face weights into a temporary HF model instance
            # then transfer state_dict. This is safer if there are slight architectural
            # differences handled by from_pretrained that aren't in a direct config init.
            temp_hf_model = HFWav2Vec2ModelForLoadingWeights.from_pretrained(model_path, config=config)

            # Load the state dict from the temp HF model into your custom model instance
            # This assumes your FloatWav2VecModel has compatible named parameters
            # (which it should if it mainly overrides `forward` and inherits from HF's Wav2Vec2Model).
            missing_keys, unexpected_keys = float_wav2vec_instance.load_state_dict(temp_hf_model.state_dict(), strict=False)
            if missing_keys:
                main_logger.warning(f"During state_dict load into FloatWav2VecModel: Missing keys: {missing_keys}")
            if unexpected_keys:
                main_logger.warning(f"During state_dict load into FloatWav2VecModel: Unexpected keys: {unexpected_keys}")

            del temp_hf_model  # Free memory of the temporary model

            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

            float_wav2vec_instance.to(device)
            float_wav2vec_instance.eval()

            # --- Prepare effective_options_dict ---
            base_opt_defaults = BaseOptions()
            effective_options = {
                "sampling_rate": base_opt_defaults.sampling_rate,
                "only_last_features": base_opt_defaults.only_last_features,
                "wav2vec_sec": base_opt_defaults.wav2vec_sec,
                "fps": base_opt_defaults.fps,
                "dim_w": base_opt_defaults.dim_w,
            }
            effective_options["hf_model_hidden_size"] = config.hidden_size  # Actual hidden size of loaded model
            if hasattr(feature_extractor, 'sampling_rate') and feature_extractor.sampling_rate:
                effective_options["sampling_rate"] = feature_extractor.sampling_rate

            if advanced_float_options:
                options_to_merge = deepcopy(advanced_float_options)
                effective_options.update(options_to_merge)
                main_logger.info(f"Applied advanced_float_options: {options_to_merge}")

            main_logger.info(f"Effective options for loaded FloatWav2VecModel: {effective_options}")

            # Pipe now contains your custom model instance
            return ((float_wav2vec_instance, feature_extractor, effective_options),)
        except Exception as e:
            main_logger.error(f"Error loading Wav2Vec model from {model_path} into FloatWav2VecModel: {e}")
            import traceback
            main_logger.error(traceback.format_exc())
            raise


class FloatAudioPreprocessAndFeatureExtract:
    UNIQUE_NAME = "FloatAudioPreprocessAndFeatureExtract"
    DISPLAY_NAME = "FLOAT Audio Feature Extract"  # Slightly shorter name

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),  # Expects pre-processed mono audio at correct SR
                "wav2vec_pipe": ("WAV2VEC_PIPE",),  # (model, feature_extractor, effective_options)
                "target_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR", "INT", "TORCH_TENSOR", "ADV_FLOAT_DICT", "WAV2VEC_PIPE")
    RETURN_NAMES = ("wav2vec_features", "audio_num_frames", "input_audio_batch_for_emotion", "effective_options",
                    "wav2vec_pipe_out")
    FUNCTION = "extract_features_with_custom_model"
    CATEGORY = BASE_CATEGORY

    def extract_features_with_custom_model(self, audio: Dict, wav2vec_pipe: tuple, target_fps: float):
        if not isinstance(wav2vec_pipe, tuple) or len(wav2vec_pipe) != 3:
            raise TypeError("wav2vec_pipe is not in the expected format (model, feature_extractor, options_dict).")

        float_wav2vec_model_instance, feature_extractor, effective_options = wav2vec_pipe

        expected_sr = effective_options.get("sampling_rate", 16000)
        only_last_features = effective_options.get("only_last_features", False)  # Align with BaseOptions default

        current_rank_device = float_wav2vec_model_instance.device

        # --- Input Audio Batch Handling & Validation (same as before) ---
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise TypeError("Input 'audio' must be a ComfyUI AUDIO dictionary.")
        input_waveform_batch = audio["waveform"]
        input_sample_rate = audio["sample_rate"]
        if input_sample_rate != expected_sr:
            raise ValueError(f"Input audio SR ({input_sample_rate} Hz) != expected SR ({expected_sr} Hz). Resample upstream.")
        if input_waveform_batch.ndim == 2:
            input_waveform_batch = input_waveform_batch.unsqueeze(1)
        elif input_waveform_batch.ndim == 3:
            if input_waveform_batch.shape[1] != 1:
                raise ValueError("Input audio must be mono.")
        else:
            raise ValueError("audio['waveform'] must be 2D or 3D.")
        batch_size = input_waveform_batch.shape[0]
        waveform_for_extractor_input = input_waveform_batch.squeeze(1).cpu().numpy()

        if batch_size == 1:
            input_to_extractor = waveform_for_extractor_input[0]
        else:
            input_to_extractor = [waveform_for_extractor_input[i] for i in range(batch_size)]

        processed_audio_dict = feature_extractor(input_to_extractor, sampling_rate=expected_sr, return_tensors='pt',
                                                 padding=True)
        preprocessed_audio_batched_cpu = processed_audio_dict.input_values
        if preprocessed_audio_batched_cpu.ndim == 1 and batch_size == 1:
            preprocessed_audio_batched_cpu = preprocessed_audio_batched_cpu.unsqueeze(0)
        # --- End Audio Preprocessing ---

        num_samples_after_prep = preprocessed_audio_batched_cpu.shape[1]
        audio_on_device = preprocessed_audio_batched_cpu.to(current_rank_device)

        audio_num_frames = math.ceil(num_samples_after_prep * target_fps / expected_sr)
        main_logger.info(f"Target audio_num_frames for feature extraction: {audio_num_frames}")

        # --- Inference using FloatWav2VecModel instance ---
        # Its forward(self, input_values, seq_len, ...) handles interpolation.
        wav2vec_features_gpu = None
        with torch.no_grad():
            float_wav2vec_model_instance.eval()  # Ensure eval mode
            # Call custom model's forward method.
            # The original AudioEncoder's get_wav2vec2_feature does:
            #   a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states=not self.only_last_features)
            #   if self.only_last_features: a = a.last_hidden_state
            #   else: a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3).reshape(B, T, -1)

            model_output = float_wav2vec_model_instance(
                input_values=audio_on_device,
                seq_len=audio_num_frames,  # This is used for interpolation inside the model
                output_hidden_states=not only_last_features  # Request hidden states if needed
            )

            if only_last_features:
                wav2vec_features_gpu = model_output.last_hidden_state  # (B, audio_num_frames, D_hidden)
            else:
                if model_output.hidden_states is None or len(model_output.hidden_states) <= 1:
                    raise ValueError("Requested all hidden states (only_last_features=False) but model did not return them "
                                     "or not enough layers.")
                # hidden_states is a tuple of tensors (B, interpolated_T, D_hidden) for each layer
                # The first element (index 0) is usually the input embeddings.
                # Original AudioEncoder stacks from hidden_states[1:].
                # (B, T_interpolated, NumLayers, D_hidden)
                stacked_hidden_states = torch.stack(model_output.hidden_states[1:], dim=2)
                wav2vec_features_gpu = stacked_hidden_states.reshape(
                    batch_size,
                    audio_num_frames,  # T_interpolated should match audio_num_frames
                    -1  # NumLayers * D_hidden
                )

        main_logger.debug(f"Extracted Wav2Vec features with custom model: shape {wav2vec_features_gpu.shape}")

        return (wav2vec_features_gpu.cpu(),
                audio_num_frames,
                preprocessed_audio_batched_cpu.cpu(),
                effective_options,
                wav2vec_pipe)


class LoadAudioProjectionLayer:
    UNIQUE_NAME = "LoadAudioProjectionLayer"
    DISPLAY_NAME = "Load Audio Projection Layer"

    @classmethod
    def INPUT_TYPES(cls):
        proj_models_path = os.path.join(folder_paths.models_dir, PROJECTIONS_DIR)
        if not os.path.isdir(proj_models_path):
            try:
                os.makedirs(proj_models_path, exist_ok=True)
            except OSError:
                pass  # If it fails, the list will be empty.

        projection_files = ["No projection files found"]
        if os.path.isdir(proj_models_path):
            files = [f for f in os.listdir(proj_models_path) if f.endswith(".safetensors")]
            if files:
                projection_files = sorted(files)

        device_options = get_torch_device_options()
        default_device = "cuda" if "cuda" in device_options else "cpu"

        return {
            "required": {
                "projection_file": (projection_files,),
                "target_device": (device_options, {"default": default_device}),
            }
        }

    RETURN_TYPES = ("AUDIO_PROJECTION_LAYER",  # Custom type for the nn.Module
                    "INT", "INT")
    RETURN_NAMES = ("projection_layer", "inferred_input_dim", "inferred_output_dim_w")
    FUNCTION = "load_projection_layer"
    CATEGORY = BASE_CATEGORY + "/Loaders"

    def load_projection_layer(self, projection_file: str, target_device: str):

        if projection_file == "No projection files found":
            raise FileNotFoundError("No .safetensors files found for audio projection. "
                                    f"Place them in 'ComfyUI/models/{PROJECTIONS_DIR}/'.")

        weights_path = os.path.join(folder_paths.models_dir, PROJECTIONS_DIR, projection_file)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Projection weights file not found: {weights_path}")

        main_logger.info(f"Loading projection weights from {weights_path} to infer dimensions.")

        loaded_sd = None
        try:
            if hasattr(comfy.utils, 'load_torch_file'):
                # Load to CPU first
                loaded_sd = comfy.utils.load_torch_file(weights_path, safe_load=True, device=torch.device("cpu"))
            else:
                from safetensors.torch import load_file
                loaded_sd = load_file(weights_path, device="cpu")
        except Exception as e:
            main_logger.error(f"Error loading weights file {weights_path}: {e}")
            raise

        # --- Infer dimensions from the weights ---
        # Assuming the first layer is nn.Linear and its weight key is "0.weight"
        # For nn.Linear, weight shape is (out_features, in_features)
        linear_weight_key = "0.weight"
        if linear_weight_key not in loaded_sd:
            raise KeyError(f"Could not find key '{linear_weight_key}' in '{weights_path}'. "
                           "Ensure the .safetensors file contains weights for a nn.Linear layer as the first component "
                           "with keys like '0.weight' and '0.bias'.")

        linear_weight_tensor = loaded_sd[linear_weight_key]
        if linear_weight_tensor.ndim != 2:
            raise ValueError(f"Weight tensor '{linear_weight_key}' is not 2D (shape: {linear_weight_tensor.shape}). "
                             "Expected (out_features, in_features).")

        inferred_output_dim_w = linear_weight_tensor.shape[0]  # out_features
        inferred_input_feature_dim = linear_weight_tensor.shape[1]  # in_features

        main_logger.info(f"Inferred projection layer dimensions: Input={inferred_input_feature_dim}, "
                         f"Output={inferred_output_dim_w}")

        # Define the projection layer structure (matching original AudioEncoder's projection)
        projection_layer = torch.nn.Sequential(
            torch.nn.Linear(inferred_input_feature_dim, inferred_output_dim_w),
            torch.nn.LayerNorm(inferred_output_dim_w),
            torch.nn.SiLU()
        )

        # Load the state dict into the newly defined layer
        try:
            missing_keys, unexpected_keys = projection_layer.load_state_dict(loaded_sd, strict=True)
            if missing_keys:
                main_logger.warning(f"Missing keys when loading projection layer: {missing_keys}")
            if unexpected_keys:
                main_logger.warning(f"Unexpected keys when loading projection layer: {unexpected_keys}")
        except Exception as e:
            main_logger.error(f"Error loading projection weights from {weights_path}: {e}")
            raise

        device = torch.device(target_device)
        projection_layer.to(device)
        projection_layer.eval()
        main_logger.info(f"Audio projection layer loaded to {target_device} and set to eval mode.")

        return (projection_layer, inferred_input_feature_dim, inferred_output_dim_w)


class FloatApplyAudioProjection:
    UNIQUE_NAME = "FloatApplyAudioProjection"
    DISPLAY_NAME = "FLOAT Apply Audio Projection"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wav2vec_features": ("TORCH_TENSOR",),  # (B, NumFrames, D_feature_for_projection)
                "projection_layer": ("AUDIO_PROJECTION_LAYER",),  # The nn.Module from LoadAudioProjectionLayer
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR",)  # wa_latent
    RETURN_NAMES = ("wa_latent",)
    FUNCTION = "apply_projection"
    CATEGORY = BASE_CATEGORY

    def apply_projection(self, wav2vec_features: torch.Tensor, projection_layer: torch.nn.Module):

        if not isinstance(wav2vec_features, torch.Tensor):
            raise TypeError("Input 'wav2vec_features' must be a torch.Tensor.")
        if not isinstance(projection_layer, torch.nn.Module):
            raise TypeError("Input 'projection_layer' must be a torch.nn.Module.")

        # Determine the device from the projection_layer (it should already be on its target device)
        target_device = next(projection_layer.parameters()).device

        features_on_device = wav2vec_features.to(target_device)

        main_logger.info(f"Applying audio projection layer to features of shape {features_on_device.shape}.")

        # The projection layer expects (B, T, D_in) or (N, D_in) and processes the last dimension.
        # If features are (B, T, D_in), Linear layer will operate on D_in.
        with torch.no_grad():  # Ensure no gradients for this application
            projection_layer.eval()  # Ensure it's in eval mode
            wa_latent_gpu = projection_layer(features_on_device)  # (B, NumFrames, D_target_w)

        main_logger.info(f"Output wa_latent shape: {wa_latent_gpu.shape}")

        # Output on CPU as per ComfyUI convention for intermediate tensors unless specified
        return (wa_latent_gpu.cpu(),)


class LoadEmotionRecognitionModel:
    UNIQUE_NAME = "LoadEmotionRecognitionModel"
    DISPLAY_NAME = "Load Emotion Recognition Model"

    @classmethod
    def INPUT_TYPES(cls):
        # Scan a dedicated folder or the general audio models folder
        emotion_models_search_path = os.path.join(folder_paths.models_dir, "audio")  # Or "emotion_models"
        if not os.path.isdir(emotion_models_search_path):
            try:
                os.makedirs(emotion_models_search_path, exist_ok=True)
            except OSError:
                pass

        model_folders = ["No models found"]
        if os.path.isdir(emotion_models_search_path):
            folders = []
            for f_name in os.listdir(emotion_models_search_path):
                full_path = os.path.join(emotion_models_search_path, f_name)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "config.json")):
                    # Simple check, could be more specific for sequence classification models
                    folders.append(f_name)
            if folders:
                model_folders = sorted(folders)

        device_options = get_torch_device_options()
        default_device = "cuda" if "cuda" in device_options else "cpu"
        default_model = ESPR if ESPR in model_folders else model_folders[0]

        return {
            "required": {
                "model_folder": (model_folders, {"default": default_model}),
                "target_device": (device_options, {"default": default_device}),
            }
        }

    RETURN_TYPES = ("EMOTION_MODEL_PIPE",)
    RETURN_NAMES = ("emotion_model_pipe",)
    FUNCTION = "load_emotion_model"
    CATEGORY = BASE_CATEGORY + "/Loaders"

    def load_emotion_model(self, model_folder: str, target_device: str):
        from transformers import Wav2Vec2FeatureExtractor  # Standard one for Wav2Vec2 based models
        from transformers import AutoConfig
        from .models.wav2vec2_ser import Wav2Vec2ForSpeechClassification

        if model_folder == "No models found":
            raise FileNotFoundError("No emotion models found. Place Hugging Face model folders in "
                                    f"'{os.path.join(folder_paths.models_dir, 'audio')}'.")

        model_path = os.path.join(folder_paths.models_dir, "audio", model_folder)  # Adjust if using a different subfolder
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Selected model folder not found: {model_path}")

        logger.info(f"Loading Emotion Recognition model from: {model_path} to device {target_device}")
        device = torch.device(target_device)

        try:
            config = AutoConfig.from_pretrained(model_path)
            # Ensure the model loaded is for sequence classification
            # Wav2Vec2ForSpeechClassification is what Audio2Emotion uses, which inherits Wav2Vec2PreTrainedModel
            # This is a slightly different classification used by FLOAT: (different head)
            emotion_model = Wav2Vec2ForSpeechClassification.from_pretrained(model_path, config=config, local_files_only=True)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)  # Emotion models usually bundle their FE

            emotion_model.to(device)
            emotion_model.eval()

            # Store relevant config info, especially label mappings if available
            model_config_dict = {
                "num_labels": config.num_labels,
                "id2label": config.id2label if hasattr(config, 'id2label') else None,
                "label2id": config.label2id if hasattr(config, 'label2id') else None,
                "sampling_rate": feature_extractor.sampling_rate if hasattr(feature_extractor, 'sampling_rate') else 16000,
                "model_path": model_path
            }

            # emotion_model_pipe tuple: (model, feature_extractor, model_config_dict)
            return ((emotion_model, feature_extractor, model_config_dict),)
        except Exception as e:
            logger.error(f"Error loading Emotion Recognition model from {model_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


class FloatExtractEmotionWithCustomModel:
    UNIQUE_NAME = "FloatExtractEmotionWithCustomModel"
    DISPLAY_NAME = "FLOAT Extract Emotion from Features"

    @classmethod
    def INPUT_TYPES(cls):
        emotion_options = EMOTIONS
        return {
            "required": {
                "processed_audio_features": ("TORCH_TENSOR",),  # (B, NumSamplesAfterPrep) from a Wav2Vec2FeatureExtractor
                "emotion_model_pipe": ("EMOTION_MODEL_PIPE",),   # (emotion_model, fe_for_emo_model_ref_only, config)
                "emotion": (emotion_options, {"default": "none"}),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR", "EMOTION_MODEL_PIPE")
    RETURN_NAMES = ("we_latent", "emotion_model_pipe_out")
    FUNCTION = "extract_emotion_from_features"  # Renamed function
    CATEGORY = BASE_CATEGORY

    def extract_emotion_from_features(self, processed_audio_features: torch.Tensor,
                                      emotion_model_pipe: tuple,
                                      emotion: str):
        if not isinstance(emotion_model_pipe, tuple) or len(emotion_model_pipe) != 3:
            raise TypeError("emotion_model_pipe is not in the expected format (model, feature_extractor_ref, config_dict).")

        emotion_model, _feature_extractor_ref_emo, model_config_emo = emotion_model_pipe
        # The _feature_extractor_ref_emo from the pipe is mostly for reference or if this node
        # HAD to do its own FE. Since we now take processed_audio_features, it's less critical here.

        current_rank_device = emotion_model.device

        # --- Input Validation for processed_audio_features ---
        if not isinstance(processed_audio_features, torch.Tensor):
            raise TypeError("Input 'processed_audio_features' must be a torch.Tensor.")
        # Expected shape: (B, NumSamplesAfterPrep) from a Wav2Vec2FeatureExtractor
        if processed_audio_features.ndim != 2:
            raise ValueError(f"Input 'processed_audio_features' must be a 2D tensor (Batch, NumSamplesAfterPrep), "
                             f"got {processed_audio_features.ndim}D with shape {processed_audio_features.shape}.")

        batch_size = processed_audio_features.shape[0]
        audio_on_device_emo = processed_audio_features.to(current_rank_device)

        # --- Emotion Prediction or One-Hot Encoding ---
        we_latent_gpu = None
        num_labels = model_config_emo.get("num_labels")
        # id2label = model_config_emo.get("id2label") # Not strictly needed for output, but good for debug
        label2id = model_config_emo.get("label2id")

        if num_labels is None:
            raise ValueError("Number of labels (num_labels) not found in emotion model config from pipe.")

        selected_emotion_lower = str(emotion).lower()
        emo_idx = None

        # Determine emo_idx based on label2id from the loaded emotion model's config
        if label2id and selected_emotion_lower != "none":
            emo_idx = label2id.get(selected_emotion_lower)  # Default None if not found
            if emo_idx is None:
                logger.warning(f"Specified emotion '{selected_emotion_lower}' not found in loaded emotion model's "
                               "label2id map. Predicting from audio instead.")
        elif selected_emotion_lower != "none":  # No label2id map in config
            logger.warning("label2id map not available in loaded emotion model config. Cannot use specified emotion "
                           f"'{selected_emotion_lower}'. Predicting from audio.")

        with torch.no_grad():
            emotion_model.eval()
            # Predict if "none" or if specified emotion is not valid/mappable
            if selected_emotion_lower == "none" or emo_idx is None:
                if selected_emotion_lower != "none" and emo_idx is None:  # Tried to specify but failed
                    logger.info(f"Failed to map '{selected_emotion_lower}'. Predicting emotion from audio features.")
                else:  # Explicitly "none" or successfully mapped emo_idx is None (should not happen if map exists)
                    logger.info("Predicting emotion from audio features using custom model.")

                # The `emotion_model` (Wav2Vec2ForSequenceClassification) directly takes `input_values`
                # which is what `processed_audio_features` represents.
                logits = emotion_model(input_values=audio_on_device_emo).logits  # (B, NumLabels)
                scores_batch = F.softmax(logits, dim=-1)  # (B, NumLabels)
                we_latent_gpu = scores_batch.unsqueeze(1)  # (B, 1, NumLabels)
            else:  # Valid emo_idx found
                logger.info(f"Using specified emotion: {selected_emotion_lower} (index {emo_idx}) for batch "
                            f"size {batch_size}.")
                one_hot_single = F.one_hot(torch.tensor(emo_idx, device=current_rank_device),
                                           num_classes=num_labels).float()
                we_latent_gpu = one_hot_single.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 1, NumLabels)

        return (we_latent_gpu.cpu(), emotion_model_pipe)


class LoadFloatEncoderModel:
    UNIQUE_NAME = "LoadFloatEncoderModel"
    DISPLAY_NAME = "Load FLOAT Encoder"
    DEFAULT_ENCODER_FILENAME = "encoder.safetensors"
    CATEGORY = BASE_CATEGORY + "/Loaders"

    _INV_CHANNELS_MAP_TEMP = {}
    for k, v_or_list in CHANNELS_MAP.items():
        vs = v_or_list if isinstance(v_or_list, list) else [v_or_list]
        for v_item in vs:
            if v_item not in _INV_CHANNELS_MAP_TEMP:
                _INV_CHANNELS_MAP_TEMP[v_item] = []
            _INV_CHANNELS_MAP_TEMP[v_item].append(k)
    INV_CHANNELS_MAP = {v: max(sizes) for v, sizes in _INV_CHANNELS_MAP_TEMP.items()}

    @classmethod
    def INPUT_TYPES(cls):
        encoder_weights_path = os.path.join(folder_paths.models_dir, MOTION_AE_DIR)
        if not os.path.isdir(encoder_weights_path):
            try:
                os.makedirs(encoder_weights_path, exist_ok=True)
            except OSError:
                pass

        weight_files = []
        if os.path.isdir(encoder_weights_path):
            weight_files = sorted([f for f in os.listdir(encoder_weights_path) if f.endswith(".safetensors")])

        if cls.DEFAULT_ENCODER_FILENAME in weight_files:
            weight_files.remove(cls.DEFAULT_ENCODER_FILENAME)
            weight_files.insert(0, cls.DEFAULT_ENCODER_FILENAME)
        elif not weight_files:
            weight_files.append("No encoder files found")
        if not weight_files:  # Should not happen with logic above, but defensive
            weight_files = ["No encoder files found"]

        device_options = get_torch_device_options()
        default_device = "cuda" if "cuda" in device_options else "cpu"

        return {
            "required": {
                "encoder_file": (weight_files,),
                "target_device": (device_options, {"default": default_device}),
                "cudnn_benchmark": ("BOOLEAN", {"default": False}),  # Added CUDNN option here
            }
        }

    # Outputting inferred dimensions for user information / optional wiring
    RETURN_TYPES = ("FLOAT_ENCODER_MODEL", "INT", "INT", "INT")
    RETURN_NAMES = ("float_encoder", "inferred_input_size", "inferred_style_dim_w", "inferred_motion_dim_m")
    FUNCTION = "load_encoder_infer_arch"

    def load_encoder_infer_arch(self, encoder_file: str, target_device: str, cudnn_benchmark: bool):
        if encoder_file == "No encoder files found":
            raise FileNotFoundError(f"No encoder .safetensors files in 'models/{MOTION_AE_DIR}/'.")

        weights_path = os.path.join(folder_paths.models_dir, MOTION_AE_DIR, encoder_file)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Encoder weights file not found: {weights_path}")

        logger.info(f"Loading Encoder weights from {weights_path} to infer architecture.")

        loaded_sd = None
        try:
            if hasattr(comfy.utils, 'load_torch_file'):
                loaded_sd = comfy.utils.load_torch_file(weights_path, safe_load=True, device=torch.device("cpu"))
            else:
                from safetensors.torch import load_file
                loaded_sd = load_file(weights_path, device="cpu")
        except Exception as e:
            logger.error(f"Error loading weights file {weights_path}: {e}")
            raise

        inferred_input_size = None
        inferred_style_dim_w = None
        inferred_motion_dim_m = None

        try:
            # Infer motion_dim_m
            motion_dim_key = 'fc.4.weight'
            if motion_dim_key not in loaded_sd:
                raise KeyError(f"Key '{motion_dim_key}' not found for motion_dim inference.")
            inferred_motion_dim_m = loaded_sd[motion_dim_key].shape[0]

            # Infer style_dim_w
            style_dim_key = 'fc.0.weight'
            if style_dim_key not in loaded_sd:
                raise KeyError(f"Key '{style_dim_key}' not found for style_dim_w inference.")
            inferred_style_dim_w = loaded_sd[style_dim_key].shape[0]

            # Infer input_size
            first_conv_key_primary = 'net_app.convs.0.0.weight'
            first_conv_key_fallback = 'net_app.convs.0.weight'
            first_conv_actual_key = None
            if first_conv_key_primary in loaded_sd:
                first_conv_actual_key = first_conv_key_primary
            elif first_conv_key_fallback in loaded_sd:
                first_conv_actual_key = first_conv_key_fallback
            else:
                raise KeyError(f"Keys for first conv ('{first_conv_key_primary}' or '{first_conv_key_fallback}') not found.")
            out_channels_first_conv = loaded_sd[first_conv_actual_key].shape[0]

            if out_channels_first_conv not in self.INV_CHANNELS_MAP:
                raise ValueError(f"Cannot infer input_size: Out channels ({out_channels_first_conv}) "
                                 f"not in INV_CHANNELS_MAP. Map: {self.INV_CHANNELS_MAP}")
            inferred_input_size = self.INV_CHANNELS_MAP[out_channels_first_conv]

            logger.info(f"Inferred Encoder Arch: size={inferred_input_size}, "
                        f"style_dim_w={inferred_style_dim_w}, motion_dim_m={inferred_motion_dim_m}")
        except KeyError as e:
            logger.error(f"Could not infer arch from weights: {e}. Check file structure.")
            raise
        except Exception as e:
            logger.error(f"Error during arch inference: {e}")
            raise

        encoder_model = FloatEncoderModule(size=inferred_input_size, dim=inferred_style_dim_w,
                                           dim_motion=inferred_motion_dim_m)

        try:
            m, u = encoder_model.load_state_dict(loaded_sd, strict=True)
            if m:
                logger.warning(f"Encoder: Missing keys: {m}")
            if u:
                logger.warning(f"Encoder: Unexpected keys: {u}")
        except Exception as e:
            logger.error(f"Error applying weights to Encoder: {e}")
            raise

        # Store inferred and provided relevant parameters on the instance
        encoder_model.inferred_input_size = inferred_input_size
        encoder_model.inferred_style_dim_w = inferred_style_dim_w
        encoder_model.inferred_motion_dim_m = inferred_motion_dim_m
        encoder_model.cudnn_benchmark_setting = cudnn_benchmark  # Store this setting

        device = torch.device(target_device)
        encoder_model.to(device)
        encoder_model.eval()
        logger.info(f"FLOAT Encoder (inferred arch) loaded to {target_device}, eval mode. CUDNN bench: {cudnn_benchmark}")

        return (encoder_model, inferred_input_size, inferred_style_dim_w, inferred_motion_dim_m)


class ApplyFloatEncoder:
    UNIQUE_NAME = "ApplyFloatEncoder"
    DISPLAY_NAME = "Apply FLOAT Encoder"
    CATEGORY = BASE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "float_encoder": ("FLOAT_ENCODER_MODEL",),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR", "FLOAT_FEATS_DICT", "TORCH_TENSOR", "FLOAT_ENCODER_MODEL")
    RETURN_NAMES = ("s_r_latent", "s_r_feats_dict", "r_s_lambda_latent", "float_encoder_out")
    FUNCTION = "apply_encoder"

    def apply_encoder(self, ref_image: torch.Tensor, float_encoder: FloatEncoderModule):

        # Get settings from the float_encoder instance
        encoder_device = next(float_encoder.parameters()).device
        cudnn_benchmark_to_use = float_encoder.cudnn_benchmark_setting

        # For validation, get structural params from encoder instance
        # And general params like input_nc from BaseOptions defaults
        input_size = float_encoder.inferred_input_size

        # Get input_nc from BaseOptions for validation consistency with integrated pipeline
        # This assumes ApplyFloatEncoder should validate against typical FLOAT input.
        base_opts_for_validation = BaseOptions()
        input_nc = base_opts_for_validation.input_nc

        if not isinstance(ref_image, torch.Tensor):
            raise TypeError("Input 'ref_image' must be a torch.Tensor.")
        if ref_image.ndim != 4:
            raise ValueError(f"Input 'ref_image' is {ref_image.ndim}D, must be 4D (B, H, W, C).")

        _batch_size, height, width, channels = ref_image.shape
        if height != input_size or width != input_size:
            raise ValueError(f"Image size {height}x{width} does not match Encoder's inferred input_size {input_size}.")
        if channels != input_nc:
            raise ValueError(f"Image channels {channels} does not match expected input_nc {input_nc}.")

        local_comfy_pbar = None  # No pbar for this apply node for now

        # Use the simplified CUDNN context manager
        with manage_cudnn_benchmark(cudnn_benchmark_to_use, encoder_device):
            # float_encoder is already on its target_device from the loader.

            s_for_encoder = ref_image.permute(0, 3, 1, 2).contiguous()
            s_for_encoder = (s_for_encoder * 2.0) - 1.0
            s_for_encoder = s_for_encoder.to(encoder_device)

            float_encoder.eval()
            with torch.no_grad():
                s_r_batch_gpu, r_s_lambda_intermediate, s_r_feats_batch_list_gpu = \
                    float_encoder(input_source=s_for_encoder, input_target=None, h_start=None, pbar=local_comfy_pbar)

                if r_s_lambda_intermediate is None:  # Should be None if input_target is None
                    r_s_lambda_batch_gpu = float_encoder.fc(s_r_batch_gpu)
                else:  # This path should ideally not be taken with current usage
                    r_s_lambda_batch_gpu = r_s_lambda_intermediate

            s_r_latent_cpu = s_r_batch_gpu.cpu()
            r_s_lambda_latent_cpu = r_s_lambda_batch_gpu.cpu()
            s_r_feats_cpu_list = [feat.cpu() for feat in s_r_feats_batch_list_gpu]
            s_r_feats_dict_cpu = {"value": s_r_feats_cpu_list}

            return (s_r_latent_cpu, s_r_feats_dict_cpu, r_s_lambda_latent_cpu, float_encoder)


class LoadFloatSynthesisModel:
    UNIQUE_NAME = "LoadFloatSynthesisModel"
    DISPLAY_NAME = "Load FLOAT Synthesis"
    DEFAULT_SYNTHESIS_FILENAME = "decoder.safetensors"  # Corrected to decoder
    CATEGORY = BASE_CATEGORY + "/Loaders"

    _INV_CHANNELS_MAP_TEMP = {}  # Same INV_CHANNELS_MAP as for Encoder
    for k, v_or_list in CHANNELS_MAP.items():
        vs = v_or_list if isinstance(v_or_list, list) else [v_or_list]
        for v_item in vs:
            if v_item not in _INV_CHANNELS_MAP_TEMP:
                _INV_CHANNELS_MAP_TEMP[v_item] = []
            _INV_CHANNELS_MAP_TEMP[v_item].append(k)
    INV_CHANNELS_MAP = {v: max(sizes) for v, sizes in _INV_CHANNELS_MAP_TEMP.items()}

    @classmethod
    def INPUT_TYPES(cls):
        synth_weights_path = os.path.join(folder_paths.models_dir, MOTION_AE_DIR)
        if not os.path.isdir(synth_weights_path):
            try:
                os.makedirs(synth_weights_path, exist_ok=True)
            except OSError:
                pass

        weight_files = []  # Logic for file listing (same as LoadFloatEncoderModel)
        if os.path.isdir(synth_weights_path):
            weight_files = sorted([f for f in os.listdir(synth_weights_path) if f.endswith(".safetensors")])
        if cls.DEFAULT_SYNTHESIS_FILENAME in weight_files:
            weight_files.remove(cls.DEFAULT_SYNTHESIS_FILENAME)
            weight_files.insert(0, cls.DEFAULT_SYNTHESIS_FILENAME)
        elif not weight_files:
            weight_files.append("No synthesis files found")
        if not weight_files:
            weight_files = ["No synthesis files found"]

        device_options = get_torch_device_options()
        default_device = "cuda" if "cuda" in device_options else "cpu"

        # Defaults for required architectural choices not easily inferred
        from .options.base_options import BaseOptions  # For these defaults
        base_opts = BaseOptions()

        return {
            "required": {
                "synthesis_file": (weight_files,),
                "target_device": (device_options, {"default": default_device}),
                "channel_multiplier": ("INT", {"default": base_opts.channel_multiplier
                                       if hasattr(base_opts, 'channel_multiplier') else 1, "min": 1, "max": 8}),
                "blur_kernel_str": ("STRING", {"default": str(base_opts.blur_kernel
                                    if hasattr(base_opts, 'blur_kernel') else [1, 3, 3, 1])}),  # e.g., "[1,3,3,1]"
                "cudnn_benchmark": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FLOAT_SYNTHESIS_MODEL", "INT", "INT", "INT")
    RETURN_NAMES = ("float_synthesis", "inferred_size", "inferred_style_dim", "inferred_motion_dim")
    FUNCTION = "load_synthesis_infer_arch"

    def load_synthesis_infer_arch(self, synthesis_file: str, target_device: str,
                                  channel_multiplier: int, blur_kernel_str: str,
                                  cudnn_benchmark: bool):
        if synthesis_file == "No synthesis files found":
            raise FileNotFoundError(f"No synthesis .safetensors files in 'models/{MOTION_AE_DIR}/'.")

        weights_path = os.path.join(folder_paths.models_dir, MOTION_AE_DIR, synthesis_file)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Synthesis weights file not found: {weights_path}")

        try:
            blur_kernel = eval(blur_kernel_str)  # Evaluate string to list
            if not (isinstance(blur_kernel, list) and all(isinstance(x, int) for x in blur_kernel)):
                raise ValueError("Blur kernel must be a list of integers (e.g., '[1,3,3,1]')")
        except Exception as e:
            raise ValueError(f"Invalid blur_kernel_str format: {e}. Must be Python list syntax e.g. '[1,3,3,1]'")

        logger.info(f"Loading Synthesis weights from {weights_path} to CPU to infer architecture.")
        loaded_sd = None
        try:
            cpu_device = torch.device("cpu")
            if hasattr(comfy.utils, 'load_torch_file'):
                loaded_sd = comfy.utils.load_torch_file(weights_path, safe_load=True, device=cpu_device)
            else:
                from safetensors.torch import load_file
                loaded_sd = load_file(weights_path, device="cpu")
        except Exception as e:
            logger.error(f"Error loading weights file {weights_path}: {e}")
            raise

        inferred_size = None
        inferred_style_dim = None
        inferred_motion_dim = None

        try:
            # 1. Infer style_dim (from conv1.conv.modulation.weight)
            #    Key: 'conv1.conv.modulation.weight', Shape: (in_channel_of_conv1, style_dim)
            #    The EqualLinear for modulation has weight (in_channel_of_mod_conv, style_dim)
            #    No, it's (in_channel_of_conv1, style_dim_of_modulation_network=style_dim)
            #    The modulation layer itself: EqualLinear(style_dim, in_channel, bias_init=1)
            #    Weight shape: (in_channel, style_dim)
            style_dim_key = 'conv1.conv.modulation.weight'  # Modulation of the first StyledConv
            if style_dim_key not in loaded_sd:
                raise KeyError(f"Key '{style_dim_key}' for style_dim inference not found.")
            inferred_style_dim = loaded_sd[style_dim_key].shape[1]  # style_dim is the second dimension

            # 2. Infer motion_dim (from direction.weight)
            #    Key: 'direction.weight', Shape: (style_dim_for_qr_maybe_512, motion_dim)
            motion_dim_key = 'direction.weight'
            if motion_dim_key not in loaded_sd:
                raise KeyError(f"Key '{motion_dim_key}' for motion_dim inference not found.")
            inferred_motion_dim = loaded_sd[motion_dim_key].shape[1]

            # 3. Infer size (from input.input constant parameter's channel dimension)
            #    Key: 'input.input', Shape: (1, channels[4], 4, 4)
            #    The channel dimension is channels[4], which is fixed at 512 in the CHANNELS_MAP.
            #    This means channels[4] = 512 for the ConstantInput.
            #    The `size` parameter for Synthesis determines log_size and thus number of layers.
            #    The number of layers (e.g., in self.convs) is (log_size - 2) * 2.
            #    Example: size=512 -> log_size=9. num_conv_pairs = 7. Total convs items = 14.
            #    We can count the number of 'convs.X...' keys or 'to_rgbs.X...' keys.
            num_torgb_layers = 0
            for k in loaded_sd.keys():
                if k.startswith("to_rgbs.") and k.endswith(".conv.0.weight"):  # Counting distinct ToRGB blocks
                    # Key example: to_rgbs.0.conv.0.weight, to_rgbs.1.conv.0.weight ...
                    # The index of to_rgbs goes from 0 to (log_size - 3)
                    layer_idx_str = k.split('.')[1]
                    if layer_idx_str.isdigit():
                        num_torgb_layers = max(num_torgb_layers, int(layer_idx_str) + 1)

            if num_torgb_layers == 0:
                raise ValueError("Could not determine number of to_rgb layers to infer size.")

            # num_torgb_layers = (log_size - 2) (since to_rgbs list is for i in range(3, log_size + 1))
            # Example: size=512, log_size=9. Range(3,10) -> i=3,4,5,6,7,8,9. Length is 7.
            # So, num_torgb_layers = log_size - 2.
            # log_size = num_torgb_layers + 2
            inferred_log_size = num_torgb_layers + 2
            inferred_size = 2 ** inferred_log_size

            logger.info(f"Inferred Synthesis Architecture: size={inferred_size}, "
                        f"style_dim={inferred_style_dim}, motion_dim={inferred_motion_dim}")

        except KeyError as e:
            logger.error(f"Could not infer architecture from Synthesis weights: {e}.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Synthesis architecture inference: {e}")
            raise

        synthesis_model = FloatSynthesisModule(
            size=inferred_size,
            style_dim=inferred_style_dim,
            motion_dim=inferred_motion_dim,
            channel_multiplier=channel_multiplier,  # From user input
            blur_kernel=blur_kernel                 # From user input
        )

        try:
            m, u = synthesis_model.load_state_dict(loaded_sd, strict=True)
            if m:
                logger.warning(f"Synthesis: Missing keys: {m}")
            if u:
                logger.warning(f"Synthesis: Unexpected keys: {u}")
        except Exception as e:
            logger.error(f"Error applying loaded weights to Synthesis: {e}")
            raise

        synthesis_model.inferred_size = inferred_size
        synthesis_model.inferred_style_dim = inferred_style_dim
        synthesis_model.inferred_motion_dim = inferred_motion_dim
        synthesis_model.channel_multiplier_setting = channel_multiplier  # Store user-set hyperparams
        synthesis_model.blur_kernel_setting = blur_kernel
        synthesis_model.cudnn_benchmark_setting = cudnn_benchmark

        device = torch.device(target_device)
        synthesis_model.to(device)
        synthesis_model.eval()
        logger.info(f"FLOAT Synthesis (inferred arch) loaded to {target_device}, eval mode. CUDNN bench: {cudnn_benchmark}")

        return (synthesis_model, inferred_size, inferred_style_dim, inferred_motion_dim)


class ApplyFloatSynthesis:
    UNIQUE_NAME = "ApplyFloatSynthesis"
    DISPLAY_NAME = "Apply FLOAT Synthesis"
    CATEGORY = BASE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s_r_latent": ("TORCH_TENSOR",),
                "s_r_feats_dict": ("FLOAT_FEATS_DICT",),
                "r_d_latents": ("TORCH_TENSOR",),
                "float_synthesis": ("FLOAT_SYNTHESIS_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT_SYNTHESIS_MODEL")
    RETURN_NAMES = ("images", "float_synthesis_out")
    FUNCTION = "apply_synthesis"

    def apply_synthesis(self, s_r_latent: torch.Tensor,
                        s_r_feats_dict: Dict[str, List[torch.Tensor]],
                        r_d_latents: torch.Tensor,
                        float_synthesis: FloatSynthesisModule):

        synthesis_device = next(float_synthesis.parameters()).device
        cudnn_benchmark_to_use = float_synthesis.cudnn_benchmark_setting

        # Get structural params for validation if needed, and general defaults
        # input_size for output shape validation
        # synthesis_input_size = float_synthesis.inferred_size # Stored by loader
        # For creating empty image tensor if T=0:
        from .options.base_options import BaseOptions
        base_opts_for_validation = BaseOptions()
        output_img_size = getattr(float_synthesis, 'inferred_size', base_opts_for_validation.input_size)
        output_img_nc = base_opts_for_validation.input_nc  # Usually 3

        # --- Input Validation ---
        if not all(isinstance(t, torch.Tensor) for t in [s_r_latent, r_d_latents]):
            raise TypeError("s_r_latent and r_d_latents must be torch.Tensors.")
        if (not isinstance(s_r_feats_dict, dict) or "value" not in s_r_feats_dict or
           not isinstance(s_r_feats_dict["value"], list)):
            raise TypeError("s_r_feats_dict format error.")
        if not all(isinstance(t, torch.Tensor) for t in s_r_feats_dict["value"]):
            raise TypeError("s_r_feats_dict['value'] must be a list of Tensors.")

        batch_size = s_r_latent.shape[0]
        if not (r_d_latents.shape[0] == batch_size and
                all(feat.shape[0] == batch_size for feat in s_r_feats_dict["value"])):
            raise ValueError("Batch size mismatch in inputs for Synthesis.")

        num_frames_to_decode = r_d_latents.shape[1]
        if num_frames_to_decode == 0:
            logger.warning("r_d_latents has 0 frames. Returning empty image batch.")
            empty_images = torch.empty((0, output_img_size, output_img_size, output_img_nc), dtype=torch.float32, device='cpu')
            return (empty_images, float_synthesis)

        comfy_pbar_decode = comfy.utils.ProgressBar(num_frames_to_decode * batch_size)

        with manage_cudnn_benchmark(cudnn_benchmark_to_use, synthesis_device):
            s_r_dev = s_r_latent.to(synthesis_device)
            s_r_feats_dev_list = [feat.to(synthesis_device) for feat in s_r_feats_dict["value"]]
            r_d_dev = r_d_latents.to(synthesis_device)

            decoded_image_sequences_list = []
            float_synthesis.eval()

            if batch_size > 1:
                logger.info(f"Applying Synthesis for batch {batch_size}, {num_frames_to_decode} frames each.")

            for b_idx in range(batch_size):
                s_r_item = s_r_dev[b_idx].unsqueeze(0)
                s_r_feats_item_list = [feat[b_idx].unsqueeze(0) for feat in s_r_feats_dev_list]
                r_d_item_all_frames = r_d_dev[b_idx]  # (T, DimW)

                frames_for_this_item = []
                for t in range(num_frames_to_decode):
                    current_r_d_frame = r_d_item_all_frames[t].unsqueeze(0)  # (1, DimW)
                    wa_for_frame_t = s_r_item + current_r_d_frame           # (1, DimW)

                    with torch.no_grad():
                        img_t_gpu, _flow_info = float_synthesis(wa=wa_for_frame_t, alpha=None, feats=s_r_feats_item_list)

                    img_t_squeezed_gpu = img_t_gpu.squeeze(0)
                    img_t_processed = img_t_squeezed_gpu.permute(1, 2, 0)
                    img_t_processed = img_t_processed.clamp(-1, 1)
                    img_t_processed = ((img_t_processed + 1) / 2)
                    frames_for_this_item.append(img_t_processed.cpu())
                    comfy_pbar_decode.update(1)

                if frames_for_this_item:
                    decoded_image_sequences_list.append(torch.stack(frames_for_this_item, dim=0))

            if not decoded_image_sequences_list:
                final_images_output_cpu = torch.empty((0, output_img_size, output_img_size, output_img_nc),
                                                      dtype=torch.float32, device='cpu')
            else:
                final_images_output_cpu = torch.cat(decoded_image_sequences_list, dim=0)

            return (final_images_output_cpu, float_synthesis)


# TODO: Merge with non-VA version
class FloatGetIdentityReferenceVA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "r_s_lambda_latent": ("TORCH_TENSOR",),         # (B, inferred_motion_dim)
                "float_synthesis": ("FLOAT_SYNTHESIS_MODEL",),  # Loaded Synthesis module
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR", "FLOAT_SYNTHESIS_MODEL")
    RETURN_NAMES = ("r_s_latent", "float_synthesis_out")
    FUNCTION = "get_identity_reference_batch"
    CATEGORY = BASE_CATEGORY
    DESCRIPTION = "Derives the batched identity reference latent (r_s) from r_s_lambda."
    UNIQUE_NAME = "FloatGetIdentityReferenceVA"
    DISPLAY_NAME = "FLOAT Get Identity Reference"

    def get_identity_reference_batch(self, r_s_lambda_latent: torch.Tensor, float_synthesis: FloatSynthesisModule):
        synthesis_module = float_synthesis
        synthesis_device = next(synthesis_module.parameters()).device

        # --- Input Validation ---
        if not isinstance(r_s_lambda_latent, torch.Tensor):
            raise TypeError(f"Input 'r_s_lambda_latent' must be a torch.Tensor, "
                            f"got {type(r_s_lambda_latent)}")
        # r_s_lambda is (B, dim_m), so 2D
        if r_s_lambda_latent.ndim != 2:
            raise ValueError(f"Input 'r_s_lambda_latent' must be a 2D tensor (Batch, DimM), "
                             f"got {r_s_lambda_latent.ndim}D.")
        if r_s_lambda_latent.shape[1] != synthesis_module.inferred_motion_dim:
            raise ValueError(f"Dimension 1 of 'r_s_lambda_latent' should be ({synthesis_module.inferred_motion_dim}), "
                             f"got {r_s_lambda_latent.shape[1]}.")

        with manage_cudnn_benchmark(synthesis_module.cudnn_benchmark_setting, synthesis_device):
            try:
                synthesis_module.to(synthesis_device)
                r_s_lambda_dev = r_s_lambda_latent.to(synthesis_device)

                # --- Core Operation ---
                # synthesis_module.direction() should handle a batch input for r_s_lambda.
                # If r_s_lambda_dev is (B, dim_m), then r_s_latent should be (B, style_dim)
                # where style_dim is derived from the weights of the Direction layer (512 in original code).
                synthesis_module.eval()  # Ensure eval mode
                with torch.no_grad():
                    r_s_latent_batch_gpu = synthesis_module.direction(r_s_lambda_dev)

                r_s_latent_batch_cpu = r_s_latent_batch_gpu.cpu()

                return (r_s_latent_batch_cpu, synthesis_module)

            finally:
                # Offload after use if it's not the CPU (standard ComfyUI practice for GPU models)
                if synthesis_device.type != 'cpu':
                    synthesis_module.to(mm.unet_offload_device())

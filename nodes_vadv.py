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
from typing import Dict  # , NewType
# ComfyUI
import comfy.utils
import folder_paths

from .options.base_options import BaseOptions
from .utils.logger import main_logger
from .utils.torch import get_torch_device_options
from .utils.misc import EMOTIONS, NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.nodes_vadv")
PROJECTIONS_DIR = "float/audio_projections"
BASE_CATEGORY = "FLOAT/Very Advanced"
ESPR = "wav2vec-english-speech-emotion-recognition"


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
    # FUNCTION = "load_hf_wav2vec_model"
    FUNCTION = "load_float_wav2vec_model"
    CATEGORY = BASE_CATEGORY + "/Loaders"

    def load_hf_wav2vec_model(self, model_folder: str, target_device: str, advanced_float_options: Dict = None):
        from transformers import Wav2Vec2Model as HFWav2Vec2Model
        from transformers import Wav2Vec2FeatureExtractor as HFWav2Vec2FeatureExtractor
        from transformers import AutoConfig
        from copy import deepcopy

        if model_folder == "No models found in models/audio/":
            raise FileNotFoundError("No Wav2Vec models found. Please place Hugging Face model folders (e.g., "
                                    "'wav2vec2-base-960h') into the 'ComfyUI/models/audio/' directory.")

        model_path = os.path.join(folder_paths.models_dir, "audio", model_folder)
        if not os.path.isdir(model_path):
            # This case should ideally be prevented by the dropdown logic, but good to have.
            raise FileNotFoundError(f"Selected model folder not found: {model_path}")

        main_logger.info(f"Loading Hugging Face Wav2Vec model from: {model_path} to device {target_device}")

        device = torch.device(target_device)

        try:
            config = AutoConfig.from_pretrained(model_path)
            # Load the base Hugging Face model. This does NOT include custom interpolation from FloatWav2VecModel.
            wav2vec_model = HFWav2Vec2Model.from_pretrained(model_path, config=config)
            feature_extractor = HFWav2Vec2FeatureExtractor.from_pretrained(model_path)

            wav2vec_model.to(device)
            wav2vec_model.eval()

            # --- Prepare effective_options_dict ---
            # Get the default from BaseOptions itself for consistency
            base_opt_defaults = BaseOptions()  # Create a temporary instance to get defaults
            effective_options = {
                "sampling_rate": base_opt_defaults.sampling_rate,  # Std for Wav2Vec2, but feature_extractor might specify
                "only_last_features": base_opt_defaults.only_last_features,
                "wav2vec_sec": base_opt_defaults.wav2vec_sec,
                "fps": base_opt_defaults.fps,
                "dim_w": base_opt_defaults.dim_w,
            }

            # Update with model-specifics from its config
            effective_options["hf_model_hidden_size"] = config.hidden_size
            if hasattr(feature_extractor, 'sampling_rate') and feature_extractor.sampling_rate:
                effective_options["sampling_rate"] = feature_extractor.sampling_rate

            # Override with user-provided advanced_float_options
            if advanced_float_options:
                # Ensure we don't modify the input dict if it's passed around
                options_to_merge = deepcopy(advanced_float_options)
                effective_options.update(options_to_merge)
                main_logger.info(f"Applied advanced_float_options: {options_to_merge}")

            main_logger.info(f"Effective options for loaded Wav2Vec: {effective_options}")

            return ((wav2vec_model, feature_extractor, effective_options),)
        except Exception as e:
            main_logger.error(f"Error loading Wav2Vec model from {model_path}: {e}")
            import traceback
            main_logger.error(traceback.format_exc())
            raise

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
    DISPLAY_NAME = "FLOAT Audio Feature Extract (Very Advanced)"  # Slightly shorter name

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
    DISPLAY_NAME = "Load Audio Projection Layer (Very Advanced)"

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
    DISPLAY_NAME = "FLOAT Apply Audio Projection (Very Advanced)"

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
    DISPLAY_NAME = "FLOAT Extract Emotion from Features (Very Advanced)"  # Name updated

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

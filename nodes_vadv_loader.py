from collections import OrderedDict
import logging
import re
import os
import torch
from typing import Dict
# ComfyUI
import comfy.utils
import folder_paths

from .options.base_options import BaseOptions
from .utils.torch import get_torch_device_options
from .utils.misc import NODES_NAME
from .models.misc import CHANNELS_MAP
from .models.float.encoder import Encoder as FloatEncoderModule
from .models.float.styledecoder import Synthesis as FloatSynthesisModule
from .models.float.FMT import FlowMatchingTransformer

logger = logging.getLogger(f"{NODES_NAME}.nodes_vadv_loaders")
FILE_CATEGORY = "FLOAT/Very Advanced/Loaders"
SUFFIX = "(VA)"
PROJECTIONS_DIR = "float/audio_projections"
MOTION_AE_DIR = "float/motion_autoencoder"
FMT_SUBDIR = "float/fmt"
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
        }

    RETURN_TYPES = ("INT", "WAV2VEC_PIPE")  # Pipe: (model, feature_extractor, effective_options)
    RETURN_NAMES = ("sampling_rate", "wav2vec_pipe")
    FUNCTION = "load_float_wav2vec_model"
    CATEGORY = FILE_CATEGORY

    def load_float_wav2vec_model(self, model_folder: str, target_device: str, advanced_float_options: Dict = None):
        # We import HFWav2Vec2Model only to load its state_dict, not to return it.
        from transformers import Wav2Vec2Model as HFWav2Vec2ModelForLoadingWeights
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import AutoConfig
        # Import your custom model class
        from .models.wav2vec2 import Wav2VecModel as FloatWav2VecModel

        if model_folder == "No models found in models/audio/":
            raise FileNotFoundError("No Wav2Vec models found. Place Hugging Face model folders into 'ComfyUI/models/audio/'.")

        model_path = os.path.join(folder_paths.models_dir, "audio", model_folder)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Selected model folder not found: {model_path}")

        logger.info(f"Loading HF Wav2Vec weights from: {model_path} into FloatWav2VecModel, target device "
                    f"{target_device}")

        device = torch.device(target_device)

        try:
            # Load configuration
            config = AutoConfig.from_pretrained(model_path)

            # Instantiate YOUR FloatWav2VecModel with this config
            # Your FloatWav2VecModel.__init__(self, config) is used here.
            float_wav2vec_instance = FloatWav2VecModel(config)
            logger.debug(f"Instantiated custom FloatWav2VecModel with config from {model_folder}")

            # Load the pre-trained Hugging Face weights into a temporary HF model instance
            # then transfer state_dict. This is safer if there are slight architectural
            # differences handled by from_pretrained that aren't in a direct config init.
            temp_hf_model = HFWav2Vec2ModelForLoadingWeights.from_pretrained(model_path, config=config)

            # Load the state dict from the temp HF model into your custom model instance
            # This assumes your FloatWav2VecModel has compatible named parameters
            # (which it should if it mainly overrides `forward` and inherits from HF's Wav2Vec2Model).
            missing_keys, unexpected_keys = float_wav2vec_instance.load_state_dict(temp_hf_model.state_dict(), strict=False)
            if missing_keys:
                logger.warning(f"During state_dict load into FloatWav2VecModel: Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"During state_dict load into FloatWav2VecModel: Unexpected keys: {unexpected_keys}")

            del temp_hf_model  # Free memory of the temporary model

            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

            float_wav2vec_instance.target_device = device
            float_wav2vec_instance.hf_model_hidden_size = config.hidden_size  # Actual hidden size of loaded model

            try:
                sampling_rate = feature_extractor.sampling_rate
            except Exception:
                sampling_rate = BaseOptions().sampling_rate
            float_wav2vec_instance.expected_sr = sampling_rate

            # Pipe now contains your custom model instance
            return (sampling_rate, (float_wav2vec_instance, feature_extractor))
        except Exception as e:
            logger.error(f"Error loading Wav2Vec model from {model_path} into FloatWav2VecModel: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


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
    RETURN_NAMES = ("projection_layer", "inferred_input_dim", "dim_a")
    FUNCTION = "load_projection_layer"
    CATEGORY = FILE_CATEGORY

    def load_projection_layer(self, projection_file: str, target_device: str):

        if projection_file == "No projection files found":
            raise FileNotFoundError("No .safetensors files found for audio projection. "
                                    f"Place them in 'ComfyUI/models/{PROJECTIONS_DIR}/'.")

        weights_path = os.path.join(folder_paths.models_dir, PROJECTIONS_DIR, projection_file)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Projection weights file not found: {weights_path}")

        logger.info(f"Loading projection weights from {weights_path} to infer dimensions.")

        loaded_sd = None
        try:
            if hasattr(comfy.utils, 'load_torch_file'):
                # Load to CPU first
                loaded_sd = comfy.utils.load_torch_file(weights_path, safe_load=True, device=torch.device("cpu"))
            else:
                from safetensors.torch import load_file
                loaded_sd = load_file(weights_path, device="cpu")
        except Exception as e:
            logger.error(f"Error loading weights file {weights_path}: {e}")
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

        dim_a = linear_weight_tensor.shape[0]  # out_features
        inferred_input_feature_dim = linear_weight_tensor.shape[1]  # in_features

        logger.info(f"Inferred projection layer dimensions: Input={inferred_input_feature_dim}, "
                    f"Output={dim_a}")

        # Define the projection layer structure (matching original AudioEncoder's projection)
        projection_layer = torch.nn.Sequential(
            torch.nn.Linear(inferred_input_feature_dim, dim_a),
            torch.nn.LayerNorm(dim_a),
            torch.nn.SiLU()
        )

        # Load the state dict into the newly defined layer
        try:
            missing_keys, unexpected_keys = projection_layer.load_state_dict(loaded_sd, strict=True)
            if missing_keys:
                logger.warning(f"Missing keys when loading projection layer: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading projection layer: {unexpected_keys}")
        except Exception as e:
            logger.error(f"Error loading projection weights from {weights_path}: {e}")
            raise

        projection_layer.target_device = torch.device(target_device)
        logger.info(f"Audio projection layer loaded to {target_device} and set to eval mode.")

        projection_layer.inferred_input_feature_dim = inferred_input_feature_dim
        projection_layer.dim_a = dim_a

        return (projection_layer, inferred_input_feature_dim, dim_a)


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

    RETURN_TYPES = ("EMOTION_MODEL_PIPE", "INT")
    RETURN_NAMES = ("emotion_model_pipe", "dim_e")  # Dimension for the emotions, i.e. 7 emotions
    FUNCTION = "load_emotion_model"
    CATEGORY = FILE_CATEGORY

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

            emotion_model.target_device = device

            # We need to know how many emotions are available
            if not hasattr(config, "num_labels"):
                msg = "Missing `num_labels` in emotion recognition config"
                logger.error(msg)
                raise ValueError(msg)

            # Store relevant config info, especially label mappings if available
            model_config_dict = {
                "num_labels": config.num_labels,
                "id2label": config.id2label if hasattr(config, 'id2label') else None,
                "label2id": config.label2id if hasattr(config, 'label2id') else None,
                "sampling_rate": feature_extractor.sampling_rate if hasattr(feature_extractor, 'sampling_rate') else 16000,
                "model_path": model_path
            }

            # emotion_model_pipe tuple: (model, feature_extractor, model_config_dict)
            return ((emotion_model, feature_extractor, model_config_dict), config.num_labels)
        except Exception as e:
            logger.error(f"Error loading Emotion Recognition model from {model_path}: {e}")
            raise


class LoadFloatEncoderModel:
    UNIQUE_NAME = "LoadFloatEncoderModel"
    DISPLAY_NAME = "Load FLOAT Encoder"
    DEFAULT_ENCODER_FILENAME = "encoder.safetensors"
    CATEGORY = FILE_CATEGORY

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
    RETURN_NAMES = ("float_encoder", "inferred_input_size", "dim_w", "dim_m")
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
        dim_w = None
        dim_m = None

        try:
            # Infer motion_dim_m
            motion_dim_key = 'fc.4.weight'
            if motion_dim_key not in loaded_sd:
                raise KeyError(f"Key '{motion_dim_key}' not found for motion_dim inference.")
            dim_m = loaded_sd[motion_dim_key].shape[0]

            # Infer style_dim_w
            style_dim_key = 'fc.0.weight'
            if style_dim_key not in loaded_sd:
                raise KeyError(f"Key '{style_dim_key}' not found for style_dim_w inference.")
            dim_w = loaded_sd[style_dim_key].shape[0]

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
                        f"style_dim_w={dim_w}, motion_dim_m={dim_m}")
        except KeyError as e:
            logger.error(f"Could not infer arch from weights: {e}. Check file structure.")
            raise
        except Exception as e:
            logger.error(f"Error during arch inference: {e}")
            raise

        encoder_model = FloatEncoderModule(size=inferred_input_size, dim=dim_w,
                                           dim_motion=dim_m)

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
        encoder_model.dim_w = dim_w
        encoder_model.dim_m = dim_m
        encoder_model.cudnn_benchmark_setting = cudnn_benchmark  # Store this setting

        encoder_model.target_device = torch.device(target_device)
        logger.info(f"FLOAT Encoder (inferred arch) loaded to {target_device}, eval mode. CUDNN bench: {cudnn_benchmark}")

        return (encoder_model, inferred_input_size, dim_w, dim_m)


class LoadFloatSynthesisModel:
    UNIQUE_NAME = "LoadFloatSynthesisModel"
    DISPLAY_NAME = "Load FLOAT Synthesis"
    DEFAULT_SYNTHESIS_FILENAME = "decoder.safetensors"  # Corrected to decoder
    CATEGORY = FILE_CATEGORY

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
        # Store user-set hyperparams
        synthesis_model.channel_multiplier_setting = channel_multiplier
        synthesis_model.blur_kernel_setting = blur_kernel
        synthesis_model.cudnn_benchmark_setting = cudnn_benchmark

        synthesis_model.target_device = torch.device(target_device)
        logger.info(f"FLOAT Synthesis (inferred arch) loaded to {target_device}, eval mode. CUDNN bench: {cudnn_benchmark}")

        return (synthesis_model, inferred_size, inferred_style_dim, inferred_motion_dim)


class LoadFMTModel:
    UNIQUE_NAME = "LoadFMTModel"
    DISPLAY_NAME = "Load FLOAT FMT Model"
    DEFAULT_FMT_FILENAME = "fmt.safetensors"
    CATEGORY = FILE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        fmt_weights_path = os.path.join(folder_paths.models_dir, FMT_SUBDIR)
        if not os.path.isdir(fmt_weights_path):
            try:
                os.makedirs(fmt_weights_path, exist_ok=True)
            except OSError:
                pass

        weight_files = [cls.DEFAULT_FMT_FILENAME]
        found_default = False
        other_files_temp = []
        if os.path.isdir(fmt_weights_path):
            for f in os.listdir(fmt_weights_path):
                if f.endswith(".safetensors"):
                    if f == cls.DEFAULT_FMT_FILENAME:
                        found_default = True
                    else:
                        other_files_temp.append(f)
        if found_default:
            weight_files.extend(sorted(other_files_temp))
        elif other_files_temp:
            weight_files = sorted(other_files_temp)
        else:
            weight_files = ["No FMT files found"]
        if not weight_files:  # Should be redundant due to above logic
            weight_files = ["No FMT files found"]

        device_options = get_torch_device_options()
        default_device = "cuda" if "cuda" in device_options else "cpu"

        # Defaults for user inputs from BaseOptions
        base_opts = BaseOptions()

        return {
            "required": {
                "fmt_file": (weight_files,),
                "target_device": (device_options, {"default": default_device}),
                "cudnn_benchmark": ("BOOLEAN", {"default": False}),
                "dim_e": ("INT", {"default": base_opts.dim_e, "min": 1, "max": 100}),
                "num_heads": ("INT", {"default": base_opts.num_heads, "min": 1, "max": 32}),
                "attention_window": ("INT", {"default": base_opts.attention_window, "min": 1, "max": 20}),
                "num_prev_frames": ("INT", {"default": base_opts.num_prev_frames, "min": 0, "max": 100}),
                "fps": ("FLOAT", {"default": base_opts.fps, "min": 1.0, "max": 120.0, "step": 0.1}),
                "wav2vec_sec": ("FLOAT", {"default": base_opts.wav2vec_sec, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    # Outputting key inferred and used parameters
    RETURN_TYPES = ("FLOAT_FMT_MODEL", "FLOAT", "ADV_FLOAT_DICT")
    RETURN_NAMES = ("float_fmt_model", "fps", "fmt_options_out")
    FUNCTION = "load_fmt_model"

    def load_fmt_model(self, fmt_file: str, target_device: str, cudnn_benchmark: bool,
                       dim_e: int, num_heads: int, attention_window: int,
                       num_prev_frames: int, fps: float, wav2vec_sec: float):

        if fmt_file == "No FMT files found":
            raise FileNotFoundError(f"No FMT .safetensors files found in 'models/{FMT_SUBDIR}/'.")

        weights_path = os.path.join(folder_paths.models_dir, FMT_SUBDIR, fmt_file)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"FMT weights file not found: {weights_path}")

        logger.info(f"Loading FMT state_dict from {weights_path} to CPU for inference and validation.")
        try:
            cpu_device = torch.device("cpu")
            if hasattr(comfy.utils, 'load_torch_file'):
                loaded_sd = comfy.utils.load_torch_file(weights_path, safe_load=True, device=cpu_device)
            else:
                from safetensors.torch import load_file
                loaded_sd = load_file(weights_path, device="cpu")
        except Exception as e:
            logger.error(f"Error loading FMT weights file: {e}")
            raise

        # --- 1. Infer structural parameters from loaded_sd ---
        try:
            inferred_dim_h = loaded_sd['x_embedder.proj.weight'].shape[0]
            logger.debug(f"Found dim_h: {inferred_dim_h}")
            inferred_dim_w_for_x_embedder = loaded_sd['x_embedder.proj.weight'].shape[1]
            logger.debug(f"Found dim_w: {inferred_dim_w_for_x_embedder}")

            max_block_idx = -1
            r = re.compile(r".*blocks\.(\d+)\..*")
            for k in loaded_sd.keys():
                m = r.match(k)
                if m:
                    block_idx = int(m.group(1))
                    if block_idx > max_block_idx:
                        max_block_idx = block_idx
            if max_block_idx == -1:
                raise KeyError("Could not find FMT blocks to infer fmt_depth.")
            inferred_fmt_depth = max_block_idx + 1
            logger.debug(f"Found fmt_depth: {inferred_fmt_depth}")

            # MLP ratio: mlp_hidden_dim / hidden_size. mlp_hidden_dim is fc1.out_features.
            mlp_fc1_weight_shape = loaded_sd['blocks.0.mlp.fc1.weight'].shape
            inferred_mlp_ratio = mlp_fc1_weight_shape[0] / inferred_dim_h
            logger.debug(f"Found mlp_ratio: {inferred_mlp_ratio}")

            # Infer sum of (dim_w + dim_a + dim_e) for c_embedder
            c_embedder_total_input_dim = loaded_sd['c_embedder.weight'].shape[1]
            # Calculate inferred_dim_a
            inferred_dim_a = c_embedder_total_input_dim - inferred_dim_w_for_x_embedder - dim_e  # User provides dim_e
            logger.debug(f"Found dim_a: {inferred_dim_a}")
            if inferred_dim_a <= 0:
                raise ValueError(f"Inferred dim_a ({inferred_dim_a}) is not positive. Check c_embedder weights, "
                                 f"inferred_dim_w_for_x_embedder ({inferred_dim_w_for_x_embedder}), or dim_e ({dim_e}).")

            logger.info(f"Inferred FMT Params: dim_h={inferred_dim_h}, dim_w={inferred_dim_w_for_x_embedder}, "
                        f"fmt_depth={inferred_fmt_depth}, mlp_ratio={inferred_mlp_ratio:.2f}, dim_a={inferred_dim_a}")
        except KeyError as e:
            logger.error(f"Missing key in FMT state_dict for inference: {e}. File might be corrupt or not an FMT model.")
            raise
        except Exception as e:
            logger.error(f"Error during FMT parameter inference: {e}")
            raise

        # --- 2. Prepare opt_for_fmt using inferred and input values ---
        opt_for_fmt = BaseOptions()  # Start with BaseOptions defaults
        # Override with inferred and direct input values (these are authoritative for structure)
        opt_for_fmt.rank = torch.device(target_device)  # For mask creation device
        opt_for_fmt.dim_h = inferred_dim_h
        opt_for_fmt.fmt_depth = inferred_fmt_depth
        opt_for_fmt.mlp_ratio = inferred_mlp_ratio
        opt_for_fmt.dim_w = inferred_dim_w_for_x_embedder  # For x_embedder and c_embedder's 'wr' part
        opt_for_fmt.dim_a = inferred_dim_a
        opt_for_fmt.dim_e = dim_e
        opt_for_fmt.num_heads = num_heads
        opt_for_fmt.attention_window = attention_window
        opt_for_fmt.num_prev_frames = num_prev_frames
        opt_for_fmt.fps = fps
        opt_for_fmt.wav2vec_sec = wav2vec_sec

        # --- 3. Validate pos_embed and alignment_mask dimensions against current opt_for_fmt ---
        # These will be used by FMT.__init__ to size its internal pos_embed and alignment_mask
        num_frames_for_clip_from_opts = int(opt_for_fmt.wav2vec_sec * opt_for_fmt.fps)
        num_total_frames_from_opts = opt_for_fmt.num_prev_frames + num_frames_for_clip_from_opts

        if 'pos_embed' in loaded_sd:
            num_total_frames_from_weights = loaded_sd['pos_embed'].shape[1]
            hidden_size_from_pos_embed = loaded_sd['pos_embed'].shape[2]
            if hidden_size_from_pos_embed != opt_for_fmt.dim_h:
                raise ValueError(f"Saved 'pos_embed' hidden dim ({hidden_size_from_pos_embed}) conflicts with "
                                 f"inferred/set opt.dim_h ({opt_for_fmt.dim_h}).")
            if num_total_frames_from_weights != num_total_frames_from_opts:
                logger.warning(f"Saved 'pos_embed' is for {num_total_frames_from_weights} total frames. "
                               f"Current input options (fps, wav2vec_sec, num_prev_frames) calculate to "
                               f"{num_total_frames_from_opts} frames. "
                               "You might need to play with parameters.")
        else:
            logger.info("'pos_embed' key not found in checkpoint. FMT will initialize it based on current options.")

        # --- 4. Instantiate FMT ---
        logger.info(f"Instantiating FlowMatchingTransformer with finalized options. "
                    f"Target device for mask: {opt_for_fmt.rank}")
        # (opt_for_fmt now contains a mix of inferred structural params and user-provided hyperparams)
        fmt_model = FlowMatchingTransformer(opt=opt_for_fmt)

        # --- 5. Load weights, skipping pos_embed and alignment_mask ---
        logger.info(f"Loading weights from {weights_path} into FlowMatchingTransformer, skipping 'pos_embed' "
                    "and 'alignment_mask'.")
        keys_to_skip_loading = ['pos_embed', 'alignment_mask']
        final_sd_to_load = OrderedDict()
        skipped_keys_info = []

        for k, v in loaded_sd.items():
            if k not in keys_to_skip_loading:
                final_sd_to_load[k] = v
            else:
                skipped_keys_info.append(f"'{k}' (shape: {v.shape})")

        if skipped_keys_info:
            logger.info(f"Intentionally skipped loading from checkpoint: {', '.join(skipped_keys_info)}. "
                        "FMT will use its freshly initialized versions for these.")

        # strict=False because fmt_model has pos_embed & alignment_mask, but final_sd_to_load doesn't.
        mismatched = fmt_model.load_state_dict(final_sd_to_load, strict=False)

        # Check for other unexpected mismatches
        actual_missing_keys = [k for k in mismatched.missing_keys if k not in keys_to_skip_loading]
        if actual_missing_keys:
            logger.warning(f"FMT: Missing keys after load (excluding intentionally skipped): {actual_missing_keys}")
        if mismatched.unexpected_keys:  # Should be empty if final_sd_to_load was subset of model
            logger.warning(f"FMT: Unexpected keys from checkpoint subset during load: {mismatched.unexpected_keys}")

        # Store relevant settings on the instance
        fmt_model.final_construction_options = {k: v for k, v in vars(opt_for_fmt).items() if not k.startswith('_')}
        fmt_model.cudnn_benchmark_setting = cudnn_benchmark

        final_target_device = torch.device(target_device)
        fmt_model.target_device = final_target_device
        logger.info(f"FlowMatchingTransformer loaded to {final_target_device}, eval mode. CUDNN bench: {cudnn_benchmark}")

        # Output options dict that was used for construction
        fmt_options_out = {**vars(opt_for_fmt)}
        # Clean up non-serializable or large items from fmt_options_out if necessary, e.g. rank
        if 'rank' in fmt_options_out:
            fmt_options_out['rank'] = str(fmt_options_out['rank'])

        return (fmt_model, fps, fmt_options_out)

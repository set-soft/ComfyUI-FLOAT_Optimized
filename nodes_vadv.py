# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
import logging
import math
import torch
import torch.nn.functional as F
# from tqdm import tqdm
from typing import Dict
# ComfyUI
import comfy.utils

from .options.base_options import BaseOptions
from .utils.torch import model_to_target
from .utils.misc import EMOTIONS, NODES_NAME, TORCHDIFFEQ_FIXED_STEP_SOLVERS
from .models.float.encoder import Encoder as FloatEncoderModule
from .models.float.styledecoder import Synthesis as FloatSynthesisModule
from .models.float.FMT import FlowMatchingTransformer
from .nodes_adv import _perform_ode_sampling_loop

logger = logging.getLogger(f"{NODES_NAME}.nodes_vadv")
BASE_CATEGORY = "FLOAT/Very Advanced"
SUFFIX = "(VA)"


class FloatAudioPreprocessAndFeatureExtract:
    UNIQUE_NAME = "FloatAudioPreprocessAndFeatureExtract"
    DISPLAY_NAME = "FLOAT Audio Feature Extract"
    DESCRIPTION = ("Processes a batch of pre-validated (mono, correct SR) audio. It applies the feature extractor from the "
                   "loaded Wav2Vec pipe, runs the audio through the Wav2Vec model, and interpolates the resulting "
                   "features to match the target video FPS, making them ready for projection.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {  # Expects pre-processed mono audio at correct SR
                    "tooltip": "The raw ComfyUI audio input. Must be mono and have the correct sample rate required by "
                    "the Wav2Vec pipe."}),
                "wav2vec_pipe": ("WAV2VEC_PIPE", {  # (model, feature_extractor)
                    "tooltip": "The loaded Wav2Vec pipe, containing the model, feature extractor, and options."}),
                "target_fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "step": 0.1,
                    "tooltip": "The target video frames-per-second. Used to calculate the final number of feature frames."}),
                "only_last_features": ("BOOLEAN", {
                    "default": False,
                    "tooltip": ("If True, use only the features from the last transformer layer. "
                                "If False, concatenate features from all transformer layers, resulting in a much larger "
                                "feature dimension.")}),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR", "INT", "TORCH_TENSOR", "WAV2VEC_PIPE")
    RETURN_NAMES = ("wav2vec_features", "audio_num_frames", "processed_audio_features", "wav2vec_pipe_out")
    FUNCTION = "extract_features_with_custom_model"
    CATEGORY = BASE_CATEGORY

    def extract_features_with_custom_model(self, audio: Dict, wav2vec_pipe: tuple, target_fps: float,
                                           only_last_features: bool):
        if not isinstance(wav2vec_pipe, tuple) or len(wav2vec_pipe) != 2:
            raise TypeError("wav2vec_pipe is not in the expected format (model, feature_extractor).")

        float_wav2vec_model_instance, feature_extractor = wav2vec_pipe
        expected_sr = float_wav2vec_model_instance.expected_sr
        current_rank_device = float_wav2vec_model_instance.target_device

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
        logger.info(f"Target audio_num_frames for feature extraction: {audio_num_frames}")

        # --- Inference using FloatWav2VecModel instance ---
        # Its forward(self, input_values, seq_len, ...) handles interpolation.
        wav2vec_features_gpu = None
        with model_to_target(float_wav2vec_model_instance):
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
                    raise ValueError("Requested all hidden states (only_last_features=False) but model did "
                                     "not return them or not enough layers.")
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

        logger.debug(f"Extracted Wav2Vec features with custom model: shape {wav2vec_features_gpu.shape}")

        return (wav2vec_features_gpu.cpu(),
                audio_num_frames,
                preprocessed_audio_batched_cpu.cpu(),
                wav2vec_pipe)


class FloatApplyAudioProjection:
    UNIQUE_NAME = "FloatApplyAudioProjection"
    DISPLAY_NAME = "FLOAT Apply Audio Projection"
    DESCRIPTION = ("Applies the loaded audio projection layer to the features extracted from the Wav2Vec model. "
                   "This final step projects the high-dimensional audio features down to the motion latent space, "
                   "producing the final audio conditioning tensor (wa_latent).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wav2vec_features": ("TORCH_TENSOR", {  # (B, NumFrames, D_feature_for_projection)
                    "tooltip": "The batch of interpolated feature tensors output by the Wav2Vec feature extraction node."}),
                "projection_layer": ("AUDIO_PROJECTION_LAYER", {  # The nn.Module from LoadAudioProjectionLayer
                    "tooltip": "The loaded audio projection layer module."}),
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
        if wav2vec_features.ndim != 3:
            raise TypeError("Input 'wav2vec_features' must contain 3 dimensions")
        if wav2vec_features.shape[2] != projection_layer.inferred_input_feature_dim:
            raise TypeError("Input 'wav2vec_features' wrong size has "
                            f"{wav2vec_features.shape[2]}, expected {projection_layer.inferred_input_feature_dim}. "
                            "`only_last_features` mismatch?")

        # Determine the device from the projection_layer (it should already be on its target device)
        target_device = projection_layer.target_device

        features_on_device = wav2vec_features.to(target_device)

        logger.info(f"Applying audio projection layer to features of shape {features_on_device.shape}.")

        # The projection layer expects (B, T, D_in) or (N, D_in) and processes the last dimension.
        # If features are (B, T, D_in), Linear layer will operate on D_in.
        with model_to_target(projection_layer):
            wa_latent_gpu = projection_layer(features_on_device)  # (B, NumFrames, D_target_w)

        logger.info(f"Output wa_latent shape: {wa_latent_gpu.shape}")

        # Output on CPU as per ComfyUI convention for intermediate tensors unless specified
        return (wa_latent_gpu.cpu(),)


class FloatExtractEmotionWithCustomModel:
    UNIQUE_NAME = "FloatExtractEmotionWithCustomModel"
    DISPLAY_NAME = "FLOAT Extract Emotion from Features"
    DESCRIPTION = ("Generates the emotion conditioning latent (we). If an emotion is specified, it creates a one-hot "
                   "encoded tensor. If set to 'none', it predicts the emotion from the provided preprocessed audio "
                   "features using the loaded custom emotion recognition model.")

    @classmethod
    def INPUT_TYPES(cls):
        emotion_options = EMOTIONS
        return {
            "required": {
                "processed_audio_features": ("TORCH_TENSOR", {  # (B, NumSamplesAfterPrep) from a Wav2Vec2FeatureExtractor
                    "tooltip": "The batch of preprocessed audio features, output by a feature extractor like the one in "
                    "FloatAudioPreprocessAndFeatureExtract."}),
                "emotion_model_pipe": ("EMOTION_MODEL_PIPE", {  # (emotion_model, fe_for_emo_model_ref_only, config)
                    "tooltip": "The loaded emotion recognition model pipe."}),
                "emotion": (emotion_options, {
                    "default": "none",
                    "tooltip": "Select a specific emotion or 'none' to have the model predict the emotion from the "
                    "audio features."}),
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

        current_rank_device = emotion_model.target_device

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

        with model_to_target(emotion_model):
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


class ApplyFloatEncoder:
    UNIQUE_NAME = "ApplyFloatEncoder"
    DISPLAY_NAME = "Apply FLOAT Encoder"
    DESCRIPTION = ("Applies the loaded FLOAT Encoder to a batch of reference images. It preprocesses the images and passes "
                   "them through the encoder to extract the core appearance latent and multi-scale feature maps, "
                   "which are bundled into a single Appearance Pipe.")
    CATEGORY = BASE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE", {
                    "tooltip": "A batch of reference images, correctly sized (e.g., 512x512) for the encoder."}),
                "float_encoder": ("FLOAT_ENCODER_MODEL", {"tooltip": "The loaded FLOAT Encoder model module."}),
            }
        }

    RETURN_TYPES = ("FLOAT_APPEARANCE_PIPE", "TORCH_TENSOR", "FLOAT_ENCODER_MODEL")
    RETURN_NAMES = ("appearance_pipe (Ws→r)", "r_s_lambda_latent", "float_encoder_out")
    FUNCTION = "apply_encoder"

    def apply_encoder(self, ref_image: torch.Tensor, float_encoder: FloatEncoderModule):

        # Get settings from the float_encoder instance
        encoder_device = float_encoder.target_device

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

        with model_to_target(float_encoder):
            s_for_encoder = ref_image.permute(0, 3, 1, 2).contiguous()
            s_for_encoder = (s_for_encoder * 2.0) - 1.0
            s_for_encoder = s_for_encoder.to(encoder_device)

            s_r_batch_gpu, r_s_lambda_intermediate, s_r_feats_batch_list_gpu = \
                float_encoder(input_source=s_for_encoder, input_target=None, h_start=None, pbar=local_comfy_pbar)

            if r_s_lambda_intermediate is None:  # Should be None if input_target is None
                r_s_lambda_batch_gpu = float_encoder.fc(s_r_batch_gpu)
            else:  # This path should ideally not be taken with current usage
                r_s_lambda_batch_gpu = r_s_lambda_intermediate

        # Bundle s_r and s_r_feats into a single dictionary
        appearance_pipe = {
            "h_source": s_r_batch_gpu.cpu(),  # This is the s_r_latent, also known as wS→r
            "feats": [feat.cpu() for feat in s_r_feats_batch_list_gpu]
        }
        r_s_lambda_latent_cpu = r_s_lambda_batch_gpu.cpu()

        return (appearance_pipe, r_s_lambda_latent_cpu, float_encoder)


class ApplyFloatSynthesis:
    UNIQUE_NAME = "ApplyFloatSynthesis"
    DISPLAY_NAME = "Apply FLOAT Synthesis"
    DESCRIPTION = ("The final image generation step. It takes the bundled appearance latents (from the Appearance Pipe) and "
                   "the driven motion sequence (r_d), then uses the loaded Synthesis/Decoder model to render the "
                   "final animated image sequence frame by frame.")
    CATEGORY = BASE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "appearance_pipe": ("FLOAT_APPEARANCE_PIPE", {
                    "tooltip": "The bundled appearance information (s_r latent and feature maps) from the ApplyFloatEncoder "
                    "node. (Ws→r)"}),
                "float_synthesis": ("FLOAT_SYNTHESIS_MODEL", {
                    "tooltip": "The loaded FLOAT Synthesis (Decoder) model module."}),
                "r_d_latents": ("TORCH_TENSOR", {
                    "tooltip": "The driven motion latent sequence generated by the FMT sampler. (Wr→D)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT_SYNTHESIS_MODEL")
    RETURN_NAMES = ("images", "float_synthesis_out")
    FUNCTION = "apply_synthesis"

    def apply_synthesis(self, appearance_pipe: Dict,
                        float_synthesis: FloatSynthesisModule,
                        r_d_latents: torch.Tensor):
        # Unpack the appearance_pipe dictionary
        try:
            s_r_latent = appearance_pipe["h_source"]
            s_r_feats_list = appearance_pipe["feats"]
        except KeyError as e:
            raise KeyError(f"Input 'appearance_pipe' is missing an expected key: {e}. "
                           "It must be the output of an ApplyFloatEncoder node.")

        synthesis_device = float_synthesis.target_device

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
        if not (isinstance(s_r_feats_list, list) and all(isinstance(t, torch.Tensor) for t in s_r_feats_list)):
            raise TypeError("appearance_pipe['feats'] must be a list of Tensors.")

        batch_size = s_r_latent.shape[0]
        if not (r_d_latents.shape[0] == batch_size and
                all(feat.shape[0] == batch_size for feat in s_r_feats_list)):
            raise ValueError("Batch size mismatch in inputs for Synthesis.")

        num_frames_to_decode = r_d_latents.shape[1]
        if num_frames_to_decode == 0:
            logger.warning("r_d_latents has 0 frames. Returning empty image batch.")
            empty_images = torch.empty((0, output_img_size, output_img_size, output_img_nc), dtype=torch.float32, device='cpu')
            return (empty_images, float_synthesis)

        comfy_pbar_decode = comfy.utils.ProgressBar(num_frames_to_decode * batch_size)

        with model_to_target(float_synthesis):
            s_r_dev = s_r_latent.to(synthesis_device)
            s_r_feats_dev_list = [feat.to(synthesis_device) for feat in s_r_feats_list]
            r_d_dev = r_d_latents.to(synthesis_device)

            decoded_image_sequences_list = []

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
                "r_s_lambda_latent": ("TORCH_TENSOR", {  # (B, inferred_motion_dim)
                    "tooltip": "The motion control parameters (h_motion) output by the FLOAT Encoder."}),
                "float_synthesis": ("FLOAT_SYNTHESIS_MODEL", {  # Loaded Synthesis module
                    "tooltip": "The loaded FLOAT Synthesis (Decoder) model, which contains the 'direction' module needed for "
                    "this transformation."}),
            }
        }

    RETURN_TYPES = ("FLOAT_SYNTHESIS_MODEL", "TORCH_TENSOR")
    RETURN_NAMES = ("float_synthesis_out", "r_s_latent (Wr→s)")
    FUNCTION = "get_identity_reference_batch"
    CATEGORY = BASE_CATEGORY
    DESCRIPTION = ("Derives the identity-specific motion reference latent (r_s) from the motion control parameters "
                   "(r_s_lambda). "
                   "This node uses the `direction` module within the loaded Synthesis/Decoder model to perform "
                   "the transformation, creating a key conditioning signal for the FMT sampler.")
    UNIQUE_NAME = "FloatGetIdentityReferenceVA"
    DISPLAY_NAME = "FLOAT Get Identity Reference"

    def get_identity_reference_batch(self, r_s_lambda_latent: torch.Tensor, float_synthesis: FloatSynthesisModule):
        synthesis_module = float_synthesis
        synthesis_device = float_synthesis.target_device

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

        # Ensure model is on the correct device for THIS operation
        with model_to_target(synthesis_module):
            r_s_lambda_dev = r_s_lambda_latent.to(synthesis_device)

            # --- Core Operation ---
            # synthesis_module.direction() should handle a batch input for r_s_lambda.
            # If r_s_lambda_dev is (B, dim_m), then r_s_latent should be (B, style_dim)
            # where style_dim is derived from the weights of the Direction layer (512 in original code).
            r_s_latent_batch_gpu = synthesis_module.direction(r_s_lambda_dev)

            r_s_latent_batch_cpu = r_s_latent_batch_gpu.cpu()

            return (synthesis_module, r_s_latent_batch_cpu)


class FloatSampleMotionSequenceRD_VA:  # Changed class name slightly
    UNIQUE_NAME = "FloatSampleMotionSequenceRD_VA"
    DISPLAY_NAME = "Sample Motion Sequence RD"
    DESCRIPTION = ("The core sampling node. It uses the loaded Flow Matching Transformer (FMT) and an ODE solver to generate "
                   "the driven motion latent sequence (r_d). It takes all conditioning latents (r_s, wa, we) and "
                   "provides explicit user control over CFG scales, ODE parameters, and noise generation.")
    CATEGORY = BASE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        # Get defaults from BaseOptions for UI hints
        base_opts = BaseOptions()
        return {
            "required": {
                "r_s_latent": ("TORCH_TENSOR", {
                    "tooltip": "The reference identity latent (wr), derived from the source image. Wr→s"}),
                "wa_latent": ("TORCH_TENSOR", {
                     "tooltip": "The audio conditioning latent (wa), derived from the audio features."}),
                "audio_num_frames": ("INT", {
                    "forceInput": True,
                    "tooltip": "Total number of frames to generate, determined by the audio length and target FPS."}),
                "we_latent": ("TORCH_TENSOR", {
                     "tooltip": "The emotion conditioning latent (we), derived from emotion prediction or specification."}),
                "float_fmt_model": ("FLOAT_FMT_MODEL", {
                     "tooltip": "The loaded FlowMatchingTransformer model from a loader node."}),  # From LoadFMTModel

                # CFG Scales
                "a_cfg_scale": ("FLOAT", {
                    "default": base_opts.a_cfg_scale, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Audio Guidance Scale. Higher values make the motion follow the audio more strictly."}),
                "r_cfg_scale": ("FLOAT", {
                    "default": base_opts.r_cfg_scale, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Reference Identity Guidance Scale. (Note: Currently unused in the default FMT "
                               "implementation)."}),
                "e_cfg_scale": ("FLOAT", {
                    "default": base_opts.e_cfg_scale, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Emotion Guidance Scale. Higher values make the motion express the target emotion "
                               "more strongly."}),
                "include_r_cfg": ("BOOLEAN", {
                    "default": False,
                    "tooltip": ("Experimental! Try to include some control over the reference weight. "
                                "When enabled the `r_cfg_scale` is used.")}),

                # ODE Parameters
                "nfe": ("INT", {
                    "default": base_opts.nfe, "min": 1, "max": 1000,
                    "tooltip": "Number of Function Evaluations for the ODE solver. Higher values increase quality and "
                               "generation time."}),
                "torchdiffeq_ode_method": (TORCHDIFFEQ_FIXED_STEP_SOLVERS, {
                    "default": base_opts.torchdiffeq_ode_method,
                    "tooltip": "The specific fixed-step numerical integration method for the ODE solver."}),
                "ode_atol": ("FLOAT", {
                    "default": base_opts.ode_atol, "min": 1e-9, "max": 1e-1, "step": 1e-6, "precision": 9,
                    "tooltip": "Absolute tolerance for the ODE solver. Controls precision."}),
                "ode_rtol": ("FLOAT", {
                    "default": base_opts.ode_rtol, "min": 1e-9, "max": 1e-1, "step": 1e-6, "precision": 9,
                    "tooltip": "Relative tolerance for the ODE solver. Controls precision."}),

                # Dropout Probabilities
                "audio_dropout_prob": ("FLOAT", {
                    "default": base_opts.audio_dropout_prob, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Dropout probability for the audio condition during sampling. Set > 0 for variation."}),
                "ref_dropout_prob": ("FLOAT", {
                    "default": base_opts.ref_dropout_prob, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Dropout probability for the reference condition during sampling. Set > 0 for variation."}),
                "emotion_dropout_prob": ("FLOAT", {
                    "default": base_opts.emotion_dropout_prob, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Dropout probability for the emotion condition during sampling. Set > 0 for variation."}),

                # Seed Control
                "fix_noise_seed": ("BOOLEAN", {
                    "default": base_opts.fix_noise_seed,
                    "tooltip": "If true, the 'seed' input will be used to generate reproducible noise. "
                    "If false, behavior depends on the seed value."}),
                "seed": ("INT", {
                    "default": base_opts.seed, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "The seed for the random noise generator used by the ODE sampler."}),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR", "FLOAT_FMT_MODEL")  # Pass through model
    RETURN_NAMES = ("r_d_latents (Wr→D)", "float_fmt_model_out")
    FUNCTION = "sample_rd_sequence_va"

    def sample_rd_sequence_va(self, r_s_latent: torch.Tensor, wa_latent: torch.Tensor,
                              we_latent: torch.Tensor, audio_num_frames: int, float_fmt_model: FlowMatchingTransformer,
                              a_cfg_scale: float, r_cfg_scale: float, e_cfg_scale: float, include_r_cfg: bool,
                              nfe: int, torchdiffeq_ode_method: str, ode_atol: float, ode_rtol: float,
                              audio_dropout_prob: float, ref_dropout_prob: float, emotion_dropout_prob: float,
                              fix_noise_seed: bool, seed: int):

        # --- Get parameters from the loaded fmt_model and its construction options ---
        # fmt_model.final_construction_options should be a dict-like snapshot of the opt used to build it
        if (not hasattr(float_fmt_model, 'final_construction_options') or
           not isinstance(float_fmt_model.final_construction_options, dict)):
            logger.error("float_fmt_model does not have 'final_construction_options' dictionary. "
                         "Ensure LoadFMTModel stores this. Using BaseOptions as fallback.")
            construction_opts_source = BaseOptions()  # Fallback, might lead to issues
        else:
            # Create a temporary opt object from these stored construction options
            # This is better than modifying float_fmt_model.opt directly if it's shared
            construction_opts_source = BaseOptions()
            for k, v in float_fmt_model.final_construction_options.items():
                if hasattr(construction_opts_source, k):
                    setattr(construction_opts_source, k, v)

        model_num_prev_frames = construction_opts_source.num_prev_frames
        # num_frames_for_clip depends on fps and wav2vec_sec used during FMT construction
        model_num_frames_for_clip = int(construction_opts_source.wav2vec_sec * construction_opts_source.fps)
        model_dim_w = construction_opts_source.dim_w

        target_device_for_sampling = float_fmt_model.target_device  # Get device from model

        # --- Input Validation (similar to Advanced version) ---
        if not all(isinstance(t, torch.Tensor) for t in [r_s_latent, wa_latent, we_latent]):
            raise TypeError("All latent inputs must be torch.Tensors.")
        batch_size = wa_latent.shape[0]
        if not (r_s_latent.shape[0] == batch_size and we_latent.shape[0] == batch_size):
            raise ValueError("Batch size mismatch among r_s, wa, we latents.")
        if wa_latent.shape[1] != audio_num_frames:
            logger.warning(f"wa_latent time dim ({wa_latent.shape[1]}) != audio_num_frames ({audio_num_frames}).")

        # --- Temporarily update dropout probabilities in the opt object that FMT uses ---
        # The FMT model's forward pass reads dropout probs from its self.opt
        # We stored `final_construction_options` on the model, but `fmt_model.opt` is the live one.
        original_dropout_probs = {
            "audio": float_fmt_model.opt.audio_dropout_prob,
            "ref": float_fmt_model.opt.ref_dropout_prob,
            "emotion": float_fmt_model.opt.emotion_dropout_prob
        }
        float_fmt_model.opt.audio_dropout_prob = audio_dropout_prob
        float_fmt_model.opt.ref_dropout_prob = ref_dropout_prob
        float_fmt_model.opt.emotion_dropout_prob = emotion_dropout_prob
        logger.debug(f"Temporarily set dropout probs: audio={audio_dropout_prob}, "
                     f"ref={ref_dropout_prob}, emotion={emotion_dropout_prob}")

        # --- Handle noise generator ---
        noise_gen = None
        if fix_noise_seed:
            noise_gen = torch.Generator(target_device_for_sampling)
            # For VA node, if fix_noise_seed is True, `seed` input is the one to use.
            # The `construction_opts_source.seed` is the one from BaseOptions or advanced_opts
            # when FMT was loaded, which might be different.
            # Let's prioritize the direct `seed` input of this node when `fix_noise_seed` is True.
            final_seed_value = seed
            noise_gen.manual_seed(final_seed_value)
            logger.debug(f"VA Sampler: Using fixed seed: {final_seed_value} for ODE sampling.")
        else:  # fix_noise_seed is False
            if seed != BaseOptions().seed:  # If user provided a specific seed different from pure default
                noise_gen = torch.Generator(target_device_for_sampling)
                noise_gen.manual_seed(seed)
                logger.debug(f"VA Sampler: Using provided seed: {seed} (fix_noise_seed is False).")
            else:  # Use global RNG
                logger.debug("VA Sampler: Not using a fixed seed. Global RNG will be used by randn.")

        # Ensure input latents are on the correct device
        r_s_latent_dev = r_s_latent.to(target_device_for_sampling)
        wa_latent_dev = wa_latent.to(target_device_for_sampling)
        we_latent_dev = we_latent.to(target_device_for_sampling)

        logger.info(f"Calling ODE sampling loop for VA node, {batch_size} item(s).")
        with model_to_target(float_fmt_model):
            try:
                r_d_latents_gpu = _perform_ode_sampling_loop(
                    fmt_model=float_fmt_model,  # Pass the loaded model
                    r_s_latent_dev=r_s_latent_dev,
                    wa_latent_dev=wa_latent_dev,
                    we_latent_dev=we_latent_dev,
                    audio_num_frames=audio_num_frames,
                    model_num_prev_frames=model_num_prev_frames,
                    model_num_frames_for_clip=model_num_frames_for_clip,
                    model_dim_w=model_dim_w,
                    ode_nfe=nfe,  # From direct input
                    ode_method=torchdiffeq_ode_method,  # From direct input
                    ode_atol=ode_atol,  # From direct input
                    ode_rtol=ode_rtol,  # From direct input
                    target_device=target_device_for_sampling,
                    a_cfg_scale=a_cfg_scale,  # From direct input
                    r_cfg_scale=r_cfg_scale,  # From direct input
                    e_cfg_scale=e_cfg_scale,  # From direct input
                    include_r_cfg=include_r_cfg,  # From direct input
                    noise_seed_generator=noise_gen,  # Configured generator
                )
                r_d_latents_cpu = r_d_latents_gpu.cpu()

                # Restore original dropout probabilities on the fmt_model's opt object
                float_fmt_model.opt.audio_dropout_prob = original_dropout_probs["audio"]
                float_fmt_model.opt.ref_dropout_prob = original_dropout_probs["ref"]
                float_fmt_model.opt.emotion_dropout_prob = original_dropout_probs["emotion"]
                logger.debug("Restored original dropout probabilities on FMT model's opt.")

                return (r_d_latents_cpu, float_fmt_model)  # Pass through the model

            except Exception as e:  # Catch any exception from the loop
                # Restore dropout probabilities in case of error too
                float_fmt_model.opt.audio_dropout_prob = original_dropout_probs["audio"]
                float_fmt_model.opt.ref_dropout_prob = original_dropout_probs["ref"]
                float_fmt_model.opt.emotion_dropout_prob = original_dropout_probs["emotion"]
                logger.error(f"Error during VA ODE sampling: {e}")
                raise

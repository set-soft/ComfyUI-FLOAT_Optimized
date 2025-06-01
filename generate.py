# -*- coding: utf-8 -*-
# Copyright (c) 2025 DeepBrain AI Research
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
"""
    Inference Stage 2
"""

import cv2
from comfy.utils import ProgressBar
import face_alignment
import librosa
import logging
import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
from typing import Union, Dict

from .models.float.FLOAT import FLOAT
from .resample import comfy_audio_to_librosa_mono

logger = logging.getLogger("FLOAT_Optimized.generate")
fa_instance = None


class CustomTransform:
    # Image transformation to input size, normalized in CHW format
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, image_np):
        resized_image = cv2.resize(image_np, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img_float = resized_image.astype(np.float32)
        normalized_image = (img_float / 127.5) - 1.0
        tensor_image = torch.from_numpy(normalized_image.transpose((2, 0, 1)))
        return tensor_image


def get_face_align():
    global fa_instance
    if fa_instance is None:
        fa_instance = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    return fa_instance


def img_tensor_2_np_array(comfy_image_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a ComfyUI image tensor into an internal np.ndarray

    Args:
        comfy_image_tensor (Union[str, torch.Tensor]):
              A ComfyUI image tensor, typically in
              (B, H, W, C) or (H, W, C) format, float32, range [0,1], RGB.

    Returns:
        np.ndarray: The image as a NumPy array in (H, W, C) format,
                    uint8, range [0, 255], RGB.
    """
    # Gemini 2.5 Pro adapted to support ComfyUI images
    # Ensure tensor is on CPU
    if comfy_image_tensor.device.type != 'cpu':
        comfy_image_tensor = comfy_image_tensor.cpu()

    # Handle batch dimension and select the first image
    # ComfyUI image tensors are typically (batch, height, width, channels)
    if comfy_image_tensor.ndim == 4:
        if comfy_image_tensor.shape[0] != 1:
            logger.warning(f"Warning: Input tensor has batch_size {comfy_image_tensor.shape[0]}. "
                           "img_tensor_2_np_array is processing only the first image.")
        img_tensor_hwc = comfy_image_tensor[0]  # Shape: (H, W, C)
    elif comfy_image_tensor.ndim == 3:  # Assuming (H, W, C)
        img_tensor_hwc = comfy_image_tensor
    else:
        raise ValueError(f"Unsupported tensor ndim: {comfy_image_tensor.ndim}. "
                         "Expected 3 (H,W,C) or 4 (B,H,W,C).")

    # Convert to NumPy array
    # Tensor is assumed to be float32, range [0,1], and RGB channels
    numpy_image = img_tensor_hwc.numpy()

    # Scale from [0, 1] to [0, 255] and convert to uint8
    # Using np.clip to ensure values are strictly within [0, 255] after scaling,
    # as floating point inaccuracies might push values slightly out of [0,1].
    numpy_image_scaled_uint8 = np.clip(numpy_image * 255.0, 0, 255).astype(np.uint8)

    # The tensor is assumed to be RGB, and the target format is RGB.
    return numpy_image_scaled_uint8


@torch.no_grad()
def process_img(img: np.ndarray, input_size: int, margin: float = 1.6) -> np.ndarray:
    mult = 360. / img.shape[0]

    resized_img = cv2.resize(img, dsize=(0, 0), fx=mult, fy=mult, interpolation=cv2.INTER_AREA
                             if mult < 1. else cv2.INTER_CUBIC)
    bboxes = get_face_align().face_detector.detect_from_image(resized_img)
    if not bboxes:
        msg = "Failed to detect any face in the image, no face align performed"
        logger.warning(msg)

        my = int(img.shape[0] / 2)
        mx = int(img.shape[1] / 2)
        bs = min(mx, my)
    else:
        bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score)
                  for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
        bboxes = bboxes[0]  # Just use first bbox

        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my = int((bboxes[1] + bboxes[3]) / 2)
        mx = int((bboxes[0] + bboxes[2]) / 2)

        bs = int(max(bsy, bsx) * margin)
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx = my + bs, mx + bs      # BBox center y, bbox center x

    crop_img = img[my - bs:my + bs, mx - bs:mx + bs]
    crop_img = cv2.resize(crop_img, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA
                          if mult < 1. else cv2.INTER_CUBIC)
    return crop_img


class DataProcessor:
    def __init__(self, opt, node_root_path="."):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.input_size = opt.input_size

        # wav2vec2 audio preprocessor
        if node_root_path is not None:
            # Now load from bundled preprocessor_config.json
            preprocessor_config_dir = os.path.join(node_root_path, opt.wav2vec_config_path)
            preprocessor_config_full_path = os.path.join(preprocessor_config_dir, 'preprocessor_config.json')
            if not os.path.exists(preprocessor_config_full_path):
                msg = f"Preprocessor config not found: {preprocessor_config_full_path}"
                logger.error(msg)
                raise FileNotFoundError(msg)
            logger.debug(f"Loading Wav2Vec2FeatureExtractor from bundled config: {preprocessor_config_full_path}")
            dir_name = preprocessor_config_dir
        else:
            # From the same place where the model was loaded
            logger.debug(f"Loading Wav2Vec2FeatureExtractor from {opt.wav2vec_model_path}")
            dir_name = opt.wav2vec_model_path
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(dir_name, local_files_only=True)

        # image transform
        self.transform = CustomTransform(opt.input_size)

    def default_aud_loader(self, path: Union[str, Dict]) -> torch.Tensor:
        if isinstance(path, dict):
            # We support a native ComfyUI audio
            # Wav2Vec needs 16k sampling rate
            speech_array, sampling_rate = comfy_audio_to_librosa_mono(path['waveform'], path['sample_rate'],
                                                                      self.sampling_rate), self.sampling_rate
        else:
            speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        return self.wav2vec_preprocessor(speech_array, sampling_rate=sampling_rate, return_tensors='pt').input_values[0]

    def preprocess(self, ref_path: Union[str, torch.Tensor], audio_path: Union[str, Dict], no_crop: bool) -> dict:
        s = img_tensor_2_np_array(ref_path)
        if not no_crop:
            s = process_img(s, self.input_size, self.opt.face_margin)
        s = self.transform(s).unsqueeze(0)
        a = self.default_aud_loader(audio_path).unsqueeze(0)
        return {'s': s, 'a': a, 'p': None, 'e': None}


class InferenceAgent:
    def __init__(self, opt, node_root_path="."):
        torch.cuda.empty_cache()
        self.opt = opt
        self.rank = opt.rank
        self.node_root_path = node_root_path

        # Load Model
        self.load_model()
        self.load_weight(opt.ckpt_path, rank=self.rank)
        self.G.to(self.rank)
        self.G.eval()

        # Load Data Processor
        self.data_processor = DataProcessor(opt, node_root_path=self.node_root_path)

    def load_model(self) -> None:
        self.G = FLOAT(self.opt, node_root_path=self.node_root_path)

    def load_weight(self, checkpoint_path: str, rank: int) -> None:
        if not os.path.exists(checkpoint_path):
            msg = f"Checkpoint file not found: {checkpoint_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if self.node_root_path is not None:
            from safetensors.torch import load_file as load_safetensors
            logger.debug(f"Loading weights from safetensors file: {checkpoint_path} directly to device: {rank}")
            # Load directly to the target device if possible
            try:
                state_dict = load_safetensors(checkpoint_path, device=str(rank))
            except Exception as e:
                logger.error(f"Failed to load safetensors file '{checkpoint_path}': {e}")
                logger.info("Attempting to load to CPU first then move.")
                state_dict = load_safetensors(checkpoint_path, device="cpu")
                # If loaded to CPU, individual parameters will be moved by load_state_dict or copy_

            logger.debug(f"Applying {len(state_dict)} key-value pairs to model G.")
            try:
                missing_keys, unexpected_keys = self.G.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logger.warning(f"Missing keys in state_dict for G: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in state_dict for G: {unexpected_keys}")
                if not missing_keys and not unexpected_keys:
                    logger.debug("All keys matched successfully for model G.")
            except RuntimeError as e:
                logger.error(f"RuntimeError during G.load_state_dict: {e}")
                logger.error("This might indicate a mismatch between saved weights and model architecture.")
                raise
            finally:
                del state_dict  # Free memory
                if 'cuda' in str(rank):
                    torch.cuda.empty_cache()
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

            with torch.no_grad():
                params = []
                for model_name, model_param in self.G.named_parameters():
                    if model_name in state_dict:
                        params.append((model_name, model_param))
                    else:
                        assert "wav2vec2" in model_name  # wav2vec2 layers aren't in float.pth
                pbar = ProgressBar(len(params))
                for model_name, model_param in tqdm(params, 'Loading weights'):
                    model_param.copy_(state_dict[model_name].to(rank))
                    pbar.update(1)
            del state_dict

    @torch.no_grad()
    def run_inference(
        self,
        res_video_path: str,
        ref_path: Union[str, torch.Tensor],
        audio_path: Union[str, Dict],
        a_cfg_scale: float = 2.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = 'S2E',
        nfe: int = 10,
        no_crop: bool = False,
        seed: int = 25
    ) -> str:
        data = self.data_processor.preprocess(ref_path, audio_path, no_crop=no_crop)

        # inference
        d_hat = self.G.inference(data=data, a_cfg_scale=a_cfg_scale, r_cfg_scale=r_cfg_scale, e_cfg_scale=e_cfg_scale,
                                 emo=emo, nfe=nfe, seed=seed)
        return d_hat

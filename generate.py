"""
    Inference Stage 2
"""

import cv2
from comfy.utils import ProgressBar
import face_alignment
import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
from typing import Union, Dict

from .models.float.FLOAT import FLOAT
from .resample import comfy_audio_to_librosa_mono


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


class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.input_size = opt.input_size

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        # wav2vec2 audio preprocessor
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)

        # image transform
        self.transform = CustomTransform(opt.input_size)

    @torch.no_grad()
    def process_img(self, img:np.ndarray) -> np.ndarray:
        mult = 360. / img.shape[0]

        resized_img = cv2.resize(img, dsize=(0, 0), fx = mult, fy = mult, interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
        bboxes = bboxes[0] # Just use first bbox

        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my  = int((bboxes[1] + bboxes[3]) / 2)
        mx  = int((bboxes[0] + bboxes[2]) / 2)

        bs = int(max(bsy, bsx) * 1.6)
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx  = my + bs, mx + bs      # BBox center y, bbox center x

        crop_img = img[my - bs:my + bs,mx - bs:mx + bs]
        crop_img = cv2.resize(crop_img, dsize = (self.input_size, self.input_size), interpolation = cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        return crop_img

    def default_img_loader(self, path_or_tensor: Union[str, torch.Tensor]) -> np.ndarray:
        """
        Loads an image from a file path (str) or uses a pre-loaded ComfyUI image tensor.

        Args:
            path_or_tensor (Union[str, torch.Tensor]):
                - If str: The file path to the image.
                - If torch.Tensor: A ComfyUI image tensor, typically in
                  (B, H, W, C) or (H, W, C) format, float32, range [0,1], RGB.

        Returns:
            np.ndarray: The image as a NumPy array in (H, W, C) format,
                        uint8, range [0, 255], RGB.
        """
        # Gemini 2.5 Pro adapted to support ComfyUI images
        if isinstance(path_or_tensor, str):
            # Original file loading logic
            img = cv2.imread(path_or_tensor)
            if img is None:
                raise FileNotFoundError(f"Image not found or unable to read: {path_or_tensor}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif not isinstance(path_or_tensor, torch.Tensor):
            raise TypeError(f"Input must be a file path (str) or a torch.Tensor, got {type(path_or_tensor)}")

        # Ok, this is a ComfyUI tensor
        comfy_image_tensor = path_or_tensor

        # Ensure tensor is on CPU
        if comfy_image_tensor.device.type != 'cpu':
            comfy_image_tensor = comfy_image_tensor.cpu()

        # Handle batch dimension and select the first image
        # ComfyUI image tensors are typically (batch, height, width, channels)
        if comfy_image_tensor.ndim == 4:
            if comfy_image_tensor.shape[0] != 1:
                print(f"Warning: Input tensor has batch_size {comfy_image_tensor.shape[0]}. "
                      "default_img_loader is processing only the first image.")
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

    def default_aud_loader(self, path: Union[str, Dict]) -> torch.Tensor:
        if isinstance(path, dict):
            # We support a native ComfyUI audio
            # Wav2Vec needs 16k sampling rate
            speech_array, sampling_rate = comfy_audio_to_librosa_mono(path['waveform'], path['sample_rate'],
                                                                      self.sampling_rate), self.sampling_rate
        else:
            speech_array, sampling_rate = librosa.load(path, sr = self.sampling_rate)
        return self.wav2vec_preprocessor(speech_array, sampling_rate = sampling_rate, return_tensors = 'pt').input_values[0]


    def preprocess(self, ref_path:Union[str, torch.Tensor], audio_path:Union[str, Dict], no_crop:bool) -> dict:
        s = self.default_img_loader(ref_path)
        if not no_crop:
            s = self.process_img(s)
        s = self.transform(s).unsqueeze(0)
        a = self.default_aud_loader(audio_path).unsqueeze(0)
        return {'s': s, 'a': a, 'p': None, 'e': None}


class InferenceAgent:
    def __init__(self, opt):
        torch.cuda.empty_cache()
        self.opt = opt
        self.rank = opt.rank

        # Load Model
        self.load_model()
        self.load_weight(opt.ckpt_path, rank=self.rank)
        self.G.to(self.rank)
        self.G.eval()

        # Load Data Processor
        self.data_processor = DataProcessor(opt)

    def load_model(self) -> None:
        self.G = FLOAT(self.opt)

    def load_weight(self, checkpoint_path: str, rank: int) -> None:
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
        a_cfg_scale: float  = 2.0,
        r_cfg_scale: float  = 1.0,
        e_cfg_scale: float  = 1.0,
        emo: str            = 'S2E',
        nfe: int            = 10,
        no_crop: bool       = False,
        seed: int           = 25,
        verbose: bool       = False
    ) -> str:
        data = self.data_processor.preprocess(ref_path, audio_path, no_crop = no_crop)

        # inference
        d_hat = self.G.inference(
            data        = data,
            a_cfg_scale = a_cfg_scale,
            r_cfg_scale = r_cfg_scale,
            e_cfg_scale = e_cfg_scale,
            emo         = emo,
            nfe         = nfe,
            seed        = seed
            )
        return d_hat

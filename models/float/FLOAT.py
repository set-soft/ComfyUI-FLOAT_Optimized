# -*- coding: utf-8 -*-
# Copyright (c) 2025 DeepBrain AI Research
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
from comfy.utils import ProgressBar
import logging
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from tqdm import tqdm
from transformers import Wav2Vec2Config

from ..wav2vec2 import Wav2VecModel
from ..wav2vec2_ser import Wav2Vec2ForSpeechClassification
from ..basemodel import BaseModel
from .generator import Generator
from .FMT import FlowMatchingTransformer
from ...utils.misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.FLOAT")


# ######## Main Phase 2 model ########
class FLOAT(BaseModel):
    def __init__(self, opt, node_root_path="."):
        super().__init__()
        self.opt = opt
        self.first_run = True
        pbar = ProgressBar(4)

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)

        # motion latent auto-encoder
        self.motion_autoencoder = Generator(size=opt.input_size, style_dim=opt.dim_w, motion_dim=opt.dim_m)
        self.motion_autoencoder.requires_grad_(False)
        pbar.update(1)

        # condition encoders
        if node_root_path is not None:
            # Don't load weights, just the configs
            full_wav2vec_config_path = os.path.join(node_root_path, opt.wav2vec_config_path, 'config.json')
            full_emotion_config_path = os.path.join(node_root_path, opt.emotion_ser_config_path, 'config.json')

            if not os.path.exists(full_wav2vec_config_path):
                raise FileNotFoundError(f"Wav2Vec2 base config not found: {full_wav2vec_config_path}")
            audio_enc_config = Wav2Vec2Config.from_json_file(full_wav2vec_config_path)
            logger.info(f"Initializing AudioEncoder with config from: {full_wav2vec_config_path}")

            if not os.path.exists(full_emotion_config_path):
                raise FileNotFoundError(f"Speech Emotion Recognition config not found: {full_emotion_config_path}")
            emotion_enc_config = Wav2Vec2Config.from_json_file(full_emotion_config_path)
            # Ensure num_labels is in the config for Wav2Vec2ForSpeechClassification
            if not hasattr(emotion_enc_config, 'num_labels'):
                num_labels_from_id2label = len(emotion_enc_config.id2label) if hasattr(emotion_enc_config, 'id2label') else 7
                emotion_enc_config.num_labels = num_labels_from_id2label
                logger.debug(f"num_labels not in emotion_enc_config, set to {num_labels_from_id2label} based on id2label or "
                             "default.")
            logger.debug(f"Initializing Audio2Emotion with config from: {full_emotion_config_path}")
        else:
            # Load weights and configs
            audio_enc_config = emotion_enc_config = None
        self.audio_encoder = AudioEncoder(opt, config=audio_enc_config)
        pbar.update(1)
        self.emotion_encoder = Audio2Emotion(opt, config=emotion_enc_config)
        pbar.update(1)

        # FMT; Flow Matching Transformer
        self.fmt = FlowMatchingTransformer(opt)
        pbar.update(1)

        # ODE options
        self.odeint_kwargs = {
            'atol': self.opt.ode_atol,
            'rtol': self.opt.ode_rtol,
            'method': self.opt.torchdiffeq_ode_method
        }
        self.print_architecture(msg='Simplified architecture')
        if logger.getEffectiveLevel() < logging.DEBUG - 1:
            self.print_architecture(max_depth=-1, msg='Full architecture')

    # ######## Motion Encoder - Decoder ########
    @torch.no_grad()
    def encode_image_into_latent(self, x: torch.Tensor) -> list:
        x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None, pbar=self.pbar if self.first_run else None)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r, x_r_lambda, x_r_feats

    @torch.no_grad()
    def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
        return r_x

    @torch.no_grad()
    def decode_latent_into_image(self, s_r: torch.Tensor, s_r_feats: list, r_d: torch.Tensor) -> dict:
        # This is the original code as reference
        T = r_d.shape[1]
        d_hat = []
        for t in range(T):
            s_r_d_t = s_r + r_d[:, t]
            img_t, _ = self.motion_autoencoder.dec(s_r_d_t, alpha=None, feats=s_r_feats)
            d_hat.append(img_t)
        d_hat = torch.stack(d_hat, dim=1).squeeze()

        return {'d_hat': d_hat}

    @torch.no_grad()
    def decode_latent_into_processed_images(self, s_r: torch.Tensor, s_r_feats: list, r_d: torch.Tensor) -> torch.Tensor:
        # More efficient decoder to enable longer videos
        # Moves the frames from VRAM to RAM and arranges them in the final shape
        # Gemini 2.5 Pro transformation
        # Assumptions:
        # 1. Effective batch size (B) for decoded images is 1.
        # 2. T (number of frames from r_d.shape[1]) > 0.
        #
        # Returns:
        # - Shape: (T, H, W, C)
        # - Dtype: torch.float32 (or original float type)
        # - Range: [0, 1]
        # - Device: CPU

        T = r_d.shape[1]  # Number of frames/images, assumed > 0

        # Decode the first frame (t=0) to determine C, H, W and dtype
        # s_r and r_d[:,0] combine to form input for one "item" if s_r is (Z) or (1,Z)
        # and r_d is (T,Z) or (1,T,Z)
        s_r_d_0 = s_r + r_d[:, 0]  # Assuming this results in an effective (1, Z_dim) input to dec
        #                            or that s_r and r_d are already shaped for B=1 processing.

        # img_0_gpu will be (1, C, H, W) based on our B=1 assumption for the output of dec
        img_0_gpu, _ = self.motion_autoencoder.dec(s_r_d_0, alpha=None, feats=s_r_feats)

        _batch_dim_size, C, H, W = img_0_gpu.shape
        assert _batch_dim_size == 1, "Decoder output batch size was not 1 as assumed"

        # Pre-allocate the final output tensor on CPU.
        # Target shape for output: (T, H, W, C)
        final_images_thwc_shape = (T, H, W, C)
        processed_images_tensor_cpu = torch.empty(final_images_thwc_shape, dtype=torch.float32, device='cpu')

        # Process first frame (t=0)
        # img_0_gpu is (1, C, H, W)
        img_0_squeezed_gpu = img_0_gpu.squeeze(0)              # (C, H, W)
        img_0_processed = img_0_squeezed_gpu.permute(1, 2, 0)  # (H, W, C)
        img_0_processed = img_0_processed.detach().clamp(-1, 1)
        img_0_processed = ((img_0_processed + 1) / 2)
        processed_images_tensor_cpu[0] = img_0_processed.cpu()  # Assign to the T dimension
        self.pbar.update(1)

        # Loop through the rest of the frames (t=1 to T-1)
        for t in tqdm(range(1, T), desc="Decoding Images"):
            s_r_d_t = s_r + r_d[:, t]
            # img_t_gpu will be (1, C, H, W)
            img_t_gpu, _ = self.motion_autoencoder.dec(s_r_d_t, alpha=None, feats=s_r_feats)
            # Process current frame
            img_t_squeezed_gpu = img_t_gpu.squeeze(0)              # (C, H, W)
            img_t_processed = img_t_squeezed_gpu.permute(1, 2, 0)  # (H, W, C)
            img_t_processed = img_t_processed.detach().clamp(-1, 1)
            img_t_processed = ((img_t_processed + 1) / 2)
            processed_images_tensor_cpu[t] = img_t_processed.cpu()  # Assign to the T dimension
            self.pbar.update(1)

        return processed_images_tensor_cpu  # Shape (T, H, W, C)

    # ######## Motion Sampling and Inference ########
    @torch.no_grad()
    def sample(
        self,
        data: dict,
        a_cfg_scale: float = 1.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = None,
        nfe: int = 10,
        seed: int = None
    ) -> torch.Tensor:

        r_s, a = data['r_s'], data['a']
        B = a.shape[0]

        # make time
        time = torch.linspace(0, 1, self.opt.nfe, device=self.opt.rank)

        # encoding audio first with whole audio
        a = a.to(self.opt.rank)
        T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        wa = self.audio_encoder.inference(a, seq_len=T)

        # encoding emotion first
        emo_idx = self.emotion_encoder.label2id.get(str(emo).lower(), None)
        if emo_idx is None:
            we = self.emotion_encoder.predict_emotion(a).unsqueeze(1)
        else:
            we = F.one_hot(torch.tensor(emo_idx, device=a.device), num_classes=self.opt.dim_e).unsqueeze(0).unsqueeze(0)

        # If we want reproducible results create generator
        if self.opt.fix_noise_seed:
            g = torch.Generator(self.opt.rank)
            g.manual_seed(self.opt.seed if seed is None else seed)
        else:
            g = None

        sample = []
        # sampling chunk by chunk
        total_chunks = int(math.ceil(T / self.num_frames_for_clip))
        iterable_chunks = tqdm(range(0, total_chunks), desc="Main inference (warm-up)", disable=not self.first_run)
        sample_t = wa_t = None
        for t in iterable_chunks:
            x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.opt.rank, generator=g)

            if t == 0:  # should define the previous
                prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
                prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
            else:
                prev_x_t = sample_t[:, -self.num_prev_frames:]
                prev_wa_t = wa_t[:, -self.num_prev_frames:]

            wa_t = wa[:, t * self.num_frames_for_clip: (t+1)*self.num_frames_for_clip]

            if wa_t.shape[1] < self.num_frames_for_clip:  # padding by replicate
                wa_t = F.pad(wa_t, (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]), mode='replicate')

            def sample_chunk(tt, zt):
                out = self.fmt.forward_with_cfv(
                        t=tt.unsqueeze(0),
                        x=zt,
                        wa=wa_t,
                        wr=r_s,
                        we=we,
                        prev_x=prev_x_t,
                        prev_wa=prev_wa_t,
                        a_cfg_scale=a_cfg_scale,
                        r_cfg_scale=r_cfg_scale,
                        e_cfg_scale=e_cfg_scale
                        )

                out_current = out[:, self.num_prev_frames:]
                return out_current

            # solve ODE
            trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]
            sample.append(sample_t)
            if self.first_run:
                self.pbar.update(1)
        sample = torch.cat(sample, dim=1)[:, :T]
        return sample

    @torch.no_grad()
    def inference(
        self,
        data: dict,
        a_cfg_scale=None,
        r_cfg_scale=None,
        e_cfg_scale=None,
        emo=None,
        nfe=10,
        seed=None,
    ) -> dict:

        s, a = data['s'], data['a']

        # How many steps we will compute
        T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        steps = T  # decode steps
        if self.first_run:
            # The first run is a warm-up and is much slower
            steps += len(self.motion_autoencoder.enc.net_app.convs)  # image encode steps
            steps += int(math.ceil(T / self.num_frames_for_clip))  # sample steps
        self.pbar = ProgressBar(steps)

        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))

        if 's_r' in data:
            r_s = self.encode_identity_into_motion(s_r)
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data['r_s'] = r_s

        # set conditions
        if a_cfg_scale is None:
            a_cfg_scale = self.opt.a_cfg_scale
        if r_cfg_scale is None:
            r_cfg_scale = self.opt.r_cfg_scale
        if e_cfg_scale is None:
            e_cfg_scale = self.opt.e_cfg_scale
        sample = self.sample(data, a_cfg_scale=a_cfg_scale, r_cfg_scale=r_cfg_scale, e_cfg_scale=e_cfg_scale, emo=emo,
                             nfe=nfe, seed=seed)

        self.first_run = False

        return self.decode_latent_into_processed_images(s_r=s_r, s_r_feats=s_r_feats, r_d=sample)


#
# ################ Condition Encoders ################
#
class AudioEncoder(BaseModel):
    def __init__(self, opt, config: Wav2Vec2Config = None):
        super().__init__()
        self.opt = opt
        self.only_last_features = opt.only_last_features

        self.num_frames_for_clip = int(opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)

        if config is not None:
            logger.debug(f"AudioEncoder: Initializing Wav2VecModel with provided config (hidden_size: {config.hidden_size})")
            self.wav2vec2 = Wav2VecModel(config)  # Initialize with config
            # Determine audio_input_dim based on config and opt.only_last_features
            if opt.only_last_features:
                audio_input_dim = config.hidden_size
            else:
                # hidden_states[0] is input embeddings, so actual transformer layers are hidden_states[1:]
                # Number of transformer layers in Wav2Vec2Config is num_hidden_layers
                num_transformer_layers = config.num_hidden_layers
                audio_input_dim = num_transformer_layers * config.hidden_size
        else:
            # Load with weights
            logger.debug(f"AudioEncoder: Initializing from {opt.wav2vec_model_path}")
            # Note: we need access to the attentions
            self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only=True,
                                                         attn_implementation="eager")
            audio_input_dim = 768 if opt.only_last_features else 12 * 768
        logger.debug(f"AudioEncoder: audio_input_dim set to {audio_input_dim}")

        self.wav2vec2.feature_extractor._freeze_parameters()

        for name, param in self.wav2vec2.named_parameters():  # Freeze all wav2vec2 params
            param.requires_grad = False

        self.audio_projection = nn.Sequential(
            nn.Linear(audio_input_dim, opt.dim_w),
            nn.LayerNorm(opt.dim_w),
            nn.SiLU()
            )
        self.print_architecture()

    def get_wav2vec2_feature(self, a: torch.Tensor, seq_len: int) -> torch.Tensor:
        a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states=not self.only_last_features)
        if self.only_last_features:
            a = a.last_hidden_state
        else:
            a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
            a = a.reshape(a.shape[0], a.shape[1], -1)
        return a

    def forward(self, a: torch.Tensor, prev_a: torch.Tensor = None) -> torch.Tensor:
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)
            if a.shape[1] % int((self.num_frames_for_clip + self.num_prev_frames) *
                                self.opt.sampling_rate / self.opt.fps) != 0:
                a = F.pad(a, (0, int((self.num_frames_for_clip + self.num_prev_frames) *
                                     self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode='replicate')
            a = self.get_wav2vec2_feature(a, seq_len=self.num_frames_for_clip + self.num_prev_frames)
        else:
            if a.shape[1] % int(self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) != 0:
                a = F.pad(a, (0, int(self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) - a.shape[1]),
                          mode='replicate')
            a = self.get_wav2vec2_feature(a, seq_len=self.num_frames_for_clip)

        return self.audio_projection(a)  # frame by frame

    @torch.no_grad()
    def inference(self, a: torch.Tensor, seq_len: int) -> torch.Tensor:
        if a.shape[1] % int(seq_len * self.opt.sampling_rate / self.opt.fps) != 0:
            a = F.pad(a, (0, int(seq_len * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode='replicate')
        a = self.get_wav2vec2_feature(a, seq_len=seq_len)
        return self.audio_projection(a)


class Audio2Emotion(BaseModel):
    def __init__(self, opt, config: Wav2Vec2Config = None):
        super().__init__()
        if config is not None:
            logger.debug(f"Audio2Emotion: Initializing Wav2Vec2ForSpeechClassification with provided config "
                         f"(num_labels: {config.num_labels})")
            self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification(config)
            self.id2label = config.id2label
        else:
            logger.debug(f"Audio2Emotion: Initializing from {opt.audio2emotion_path}")
            self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(opt.audio2emotion_path,
                                                                                        local_files_only=True)
            self.id2label = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
        self.wav2vec2_for_emotion.eval()

        self.label2id = {v: k for k, v in self.id2label.items()}
        self.print_architecture()

    @torch.no_grad()
    def predict_emotion(self, a: torch.Tensor, prev_a: torch.Tensor = None) -> torch.Tensor:
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)
        logits = self.wav2vec2_for_emotion.forward(a).logits
        return F.softmax(logits, dim=1)     # scores

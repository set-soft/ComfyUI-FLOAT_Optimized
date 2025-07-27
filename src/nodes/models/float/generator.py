# -*- coding: utf-8 -*-
# Copyright (c) 2025 DeepBrain AI Research
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
from .encoder import Encoder
from .styledecoder import Synthesis

from ..basemodel import BaseModel


class Generator(BaseModel):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)
        self.print_architecture()

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img, flow = self.dec(wa, alpha, feat)
        return img

    def forward(self, img_source, img_drive, h_start=None):
        wa, alpha, feats = self.enc(img_source, img_drive, h_start)
        img_recon, flow = self.dec(wa, alpha, feats)
        return {'d_hat': img_recon, 'flow': flow}

# -*- coding: utf-8 -*-
# Copyright (c) 2025 Yuvraj Seegolam
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
from .nodes import LoadFloatModels, FloatProcess, FloatAdvancedParameters, FloatImageFaceAlign

NODE_CLASS_MAPPINGS = {
    "LoadFloatModelsOpt": LoadFloatModels,
    "FloatProcessOpt": FloatProcess,
    "FloatAdvancedParameters": FloatAdvancedParameters,
    "FloatImageFaceAlign": FloatImageFaceAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "LoadFloatModelsOpt": "Load FLOAT Models (Opt)",
     "FloatProcessOpt": "FLOAT Process (Opt)",
     "FloatAdvancedParameters": "FLOAT Advanced Options",
     "FloatImageFaceAlign": "Face Align for FLOAT"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

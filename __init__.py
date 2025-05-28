# -*- coding: utf-8 -*-
# Copyright (c) 2025 Yuvraj Seegolam
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
from .nodes import LoadFloatModels, FloatProcess, FloatAdvancedParameters

NODE_CLASS_MAPPINGS = {
    "LoadFloatModelsOpt": LoadFloatModels,
    "FloatProcessOpt": FloatProcess,
    "FloatAdvancedParameters": FloatAdvancedParameters,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "LoadFloatModelsOpt": "Load Float Models (Opt)",
     "FloatProcessOpt": "Float Process (Opt)",
     "FloatAdvancedParameters": "Float Advanced Options",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

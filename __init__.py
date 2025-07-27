# -*- coding: utf-8 -*-
# Copyright (c) 2025 Yuvraj Seegolam
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
from .src.nodes import nodes, main_logger, __version__
from .src.nodes import nodes_adv
from .src.nodes import nodes_vadv
from .src.nodes import nodes_vadv_loader
from seconohe.register_nodes import register_nodes
from seconohe import JS_PATH


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes(
    main_logger,
    [nodes, nodes_adv, nodes_vadv, nodes_vadv_loader],
    version=__version__
)
WEB_DIRECTORY = JS_PATH
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

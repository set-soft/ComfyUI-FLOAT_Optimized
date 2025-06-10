# -*- coding: utf-8 -*-
# Copyright (c) 2025 Yuvraj Seegolam
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
from . import nodes
from . import nodes_adv
from . import nodes_vadv
import inspect
import logging
from .utils.misc import NODES_NAME

init_logger = logging.getLogger(f"{NODES_NAME}.__init__")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_nodes(module):
    suffix = " " + module.SUFFIX if hasattr(module, "SUFFIX") else ""
    if suffix:
        suffix = " " + suffix
    for name, obj in inspect.getmembers(module):
        if not inspect.isclass(obj) or not hasattr(obj, "INPUT_TYPES"):
            continue
        assert hasattr(obj, "UNIQUE_NAME"), f"No name for {obj.__name__}"
        NODE_CLASS_MAPPINGS[obj.UNIQUE_NAME] = obj
        NODE_DISPLAY_NAME_MAPPINGS[obj.UNIQUE_NAME] = obj.DISPLAY_NAME + suffix


register_nodes(nodes)
register_nodes(nodes_adv)
register_nodes(nodes_vadv)

init_logger.info(f"Registering {len(NODE_CLASS_MAPPINGS)} node(s).")
init_logger.debug(f"{list(NODE_DISPLAY_NAME_MAPPINGS.values())}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

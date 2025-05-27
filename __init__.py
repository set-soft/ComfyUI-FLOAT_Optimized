from .nodes import LoadFloatModels, FloatProcess, FloatAdvancedParameters
 
NODE_CLASS_MAPPINGS = { 
    "LoadFloatModels" : LoadFloatModels,
    "FloatProcess" : FloatProcess,
    "FloatAdvancedParameters": FloatAdvancedParameters,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "LoadFloatModels" : "Load Float Models",
     "FloatProcess" : "Float Process",
     "FloatAdvancedParameters": "Float Advanced Options",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
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

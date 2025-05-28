import os
import folder_paths
import comfy.model_management as mm

from .generate import InferenceAgent
from .options.base_options import BaseOptionsJson

# List of fixed-step solvers you from torchdiffeq
TORCHDIFFEQ_FIXED_STEP_SOLVERS = [
    "euler",
    "midpoint",
    "rk4",        # Classical Runge-Kutta 4th order
    "heun2",      # Heun's method (RK2)
    "heun3",      # Heun's method (RK3)
    # "explicit_adams", # Fixed-step Adams-Bashforth (multi-step, might need more params like order)
    # "implicit_adams", # Fixed-step Adams-Moulton (multi-step, implicit)
]


class LoadFloatModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (['float.pth'],)
            },
            "optional": {
                "advanced_float_options": ("ADV_FLOAT_DICT",)
            }
        }

    RETURN_TYPES = ("FLOAT_PIPE",)
    RETURN_NAMES = ("float_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Models are auto-downloaded to /ComfyUI/models/float"

    def loadmodel(self, model, advanced_float_options=None):
        # download models if not exist
        float_models_dir = os.path.join(folder_paths.models_dir, "float")
        os.makedirs(float_models_dir, exist_ok=True)

        audio_models_dir = os.path.join(folder_paths.models_dir, "audio")

        wav2vec2_base_960h_models_dir = os.path.join(float_models_dir,"wav2vec2-base-960h")
        wav2vec_english_speech_emotion_recognition_models_dir = os.path.join(float_models_dir,"wav2vec-english-speech-emotion-recognition")

        # Allow models/audio/wav2vec2-base-960h
        if not os.path.isdir(wav2vec2_base_960h_models_dir):
            alt_dir = os.path.join(audio_models_dir, "wav2vec2-base-960h")
            if os.path.isdir(alt_dir):
                wav2vec2_base_960h_models_dir = alt_dir
                print("Using speach encoder from: "+alt_dir)
        # Allow models/audio/wav2vec-english-speech-emotion-recognition
        if not os.path.isdir(wav2vec_english_speech_emotion_recognition_models_dir):
            alt_dir = os.path.join(audio_models_dir, "wav2vec-english-speech-emotion-recognition")
            if os.path.isdir(alt_dir):
                wav2vec_english_speech_emotion_recognition_models_dir = alt_dir
                print("Using emotion decoder from: "+alt_dir)

        float_model_path = os.path.join(float_models_dir,"float.pth")

        if not os.path.exists(float_model_path) or not os.path.isdir(wav2vec2_base_960h_models_dir) or not os.path.isdir(wav2vec_english_speech_emotion_recognition_models_dir):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="yuvraj108c/float", local_dir=float_models_dir, local_dir_use_symlinks=False)

        # use custom dictionary instead of original parser for arguments
        opt = BaseOptionsJson

        if advanced_float_options is not None and isinstance(advanced_float_options, dict):
            for key, value in advanced_float_options.items():
                if hasattr(opt, key):
                    setattr(opt, key, value)

        opt.rank, opt.ngpus  = 0,1
        opt.ckpt_path = float_model_path
        opt.pretrained_dir = float_models_dir
        opt.wav2vec_model_path = wav2vec2_base_960h_models_dir
        opt.audio2emotion_path = wav2vec_english_speech_emotion_recognition_models_dir
        agent = InferenceAgent(opt)

        return (agent,)


class FloatProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "ref_audio": ("AUDIO",),
                "float_pipe": ("FLOAT_PIPE",),
                "a_cfg_scale": ("FLOAT", {"default": 2.0,"min": 1.0, "step": 0.1}),
                "e_cfg_scale": ("FLOAT", {"default": 1.0,"min": 1.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 25, "step": 1}),
                "emotion": (['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], {"default": "none"}),
                "face_align": ("BOOLEAN",{"default":False},),
                "seed": ("INT", {"default": 62064758300528, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "floatprocess"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Float Processing"

    def floatprocess(self, ref_image, ref_audio, float_pipe, a_cfg_scale, e_cfg_scale, fps, emotion, face_align, seed):
        float_pipe.G.to(float_pipe.rank)

        float_pipe.opt.fps = fps
        images_bhwc = float_pipe.run_inference(
            None,
            ref_image,
            ref_audio,
            a_cfg_scale = a_cfg_scale,
            r_cfg_scale = float_pipe.opt.r_cfg_scale,
            e_cfg_scale = e_cfg_scale,
            emo         = None if emotion == "none" else emotion,
            no_crop     = not face_align,
            seed        = seed
        )

        float_pipe.G.to(mm.unet_offload_device())

        return (images_bhwc,)


class FloatAdvancedParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "r_cfg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "step": 0.1
                }),
                "attention_window": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10, # Practical maximum
                    "step": 1,
                    "display": "number"
                }),
                "audio_dropout_prob": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "ref_dropout_prob": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "emotion_dropout_prob": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "ode_atol": ("FLOAT", { # Represent as float, can use scientific notation in input
                    "default": 1e-5,
                    "min": 1e-9,   # Practical minimum for tolerance
                    "max": 1e-1,   # Practical maximum for tolerance
                    "step": 1e-6,  # Adjust step for fine control
                    "display": "number",
                    "precision": 9 # Number of decimal places for display if needed
                }),
                "ode_rtol": ("FLOAT", {
                    "default": 1e-5,
                    "min": 1e-9,
                    "max": 1e-1,
                    "step": 1e-6,
                    "display": "number",
                    "precision": 9
                }),
                "nfe": ("INT", { # Number of Function Evaluations
                    "default": 10,
                    "min": 1,
                    "max": 1000, # Practical maximum
                    "step": 1,
                    "display": "number"
                }),
                "torchdiffeq_ode_method": (TORCHDIFFEQ_FIXED_STEP_SOLVERS, {
                    "default": "euler"
                }),
            }
        }

    RETURN_TYPES = ("ADV_FLOAT_DICT",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "get_options"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Float Advanced Options"

    def get_options(self, r_cfg_scale, attention_window, audio_dropout_prob, ref_dropout_prob, emotion_dropout_prob,
                    ode_atol, ode_rtol, nfe, torchdiffeq_ode_method):

        options_dict = {
            "r_cfg_scale": r_cfg_scale,
            "attention_window": attention_window,
            "audio_dropout_prob": audio_dropout_prob,
            "ref_dropout_prob": ref_dropout_prob,
            "emotion_dropout_prob": emotion_dropout_prob,
            "ode_atol": ode_atol,
            "ode_rtol": ode_rtol,
            "nfe": nfe,
            "torchdiffeq_ode_method": torchdiffeq_ode_method
        }

        return (options_dict,)

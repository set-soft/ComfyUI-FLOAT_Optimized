# Tool to create the unified safetensors file
# Mostly Gemini 2.5 Pro code
import torch
import os
import sys
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

# This tool should be run from the root of the nodes, i.e. python tools/save_combined_tool.py

# --- Add parent directory to sys.path to allow importing node's modules ---
# This assumes the script is in the tools/ of your ComfyUI-Float_Optimized package.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(base_dir)  # This would be custom_nodes
sys.path.insert(0, base_dir)    # Add custom_nodes/ComfyUI-Float_Optimized to path
sys.path.insert(1, parent_dir)  # Add custom_nodes to path
sys.path.insert(1, os.path.dirname(os.path.dirname(base_dir)))  # Add ComfyUI to path
# Pretend we are a package
__package__ = os.path.basename(base_dir)
__import__(__package__)

# --- Now import your classes ---
try:
    from options.base_options import BaseOptions
    from .generate import InferenceAgent      # This will import FLOAT, AudioEncoder, etc.
    # Needed for type hints if any
    # from models.wav2vec2 import Wav2VecModel
    # from models.wav2vec2_ser import Wav2Vec2ForSpeechClassification
    # from transformers import Wav2Vec2Config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure this script is run from a context where it can find 'options', 'generate', etc.")
    print(f"Current sys.path: {sys.path}")
    raise
    exit(1)


def create_and_save_combined_model(output_safetensors_path="combined_float_weights.safetensors"):
    print("Initializing BaseOptions for saving tool...")
    # We need an opt object. Create a default one.
    # The paths here will point to the *original separate model locations*
    # for this one-time saving process.
    opt = BaseOptions()

    # --- Setup paths to current separate model locations ---
    # These are where the original models should be for this script to load them.
    # This mimics the original download/setup.
    # You might need to adjust these paths if your local setup for original models differs.
    # For simplicity, let's assume a `temp_original_models` dir for this script's run.

    script_base_dir = os.path.dirname(os.path.abspath(__file__))
    # This is where snapshot_download will put things if they don't exist
    # This path should be relative to where you run this script, or absolute.
    temp_original_float_models_dir = os.path.join(script_base_dir, "temp_original_models")
    os.makedirs(temp_original_float_models_dir, exist_ok=True)

    opt.pretrained_dir = temp_original_float_models_dir  # For InferenceAgent
    opt.ckpt_path = os.path.join(temp_original_float_models_dir, "float.pth")
    opt.wav2vec_model_path = os.path.join(temp_original_float_models_dir, "wav2vec2-base-960h")
    opt.audio2emotion_path = os.path.join(temp_original_float_models_dir, "wav2vec-english-speech-emotion-recognition")

    # Ensure original models are present for this script to load from
    if not (os.path.exists(opt.ckpt_path) and
            os.path.isdir(opt.wav2vec_model_path) and
            os.path.isdir(opt.audio2emotion_path)):
        print(f"Original models not found in {temp_original_float_models_dir}. Attempting download from Hugging Face Hub...")
        try:
            snapshot_download(repo_id="yuvraj108c/float",
                              local_dir=temp_original_float_models_dir,
                              local_dir_use_symlinks=False,
                              # Ensure all necessary files are downloaded
                              allow_patterns=["*.pth", "wav2vec2-base-960h/*", "wav2vec-english-speech-emotion-recognition/*"])
            print("Download of original models complete.")
        except Exception as e:
            print(f"Failed to download original models: {e}")
            print(f"Please manually ensure 'float.pth', 'wav2vec2-base-960h', and "
                  "'wav2vec-english-speech-emotion-recognition' "
                  f"are inside '{temp_original_float_models_dir}' before running this script.")
            return

    # Device for loading (CPU is fine)
    opt.rank = torch.device("cpu")
    # opt.cudnn_benchmark_enabled is in BaseOptions, will be False by default.

    print("Initializing InferenceAgent with original model paths to load weights for saving...")
    print(f"  float.pth path: {opt.ckpt_path}")
    print(f"  wav2vec_model_path (for AudioEncoder's Wav2VecModel): {opt.wav2vec_model_path}")
    print(f"  audio2emotion_path (for Audio2Emotion's Model): {opt.audio2emotion_path}")

    # This `InferenceAgent` instantiation will trigger the original loading mechanism:
    # - FLOAT will init its components (motion_autoencoder, fmt, audio_encoder, emotion_encoder)
    # - audio_encoder will init its wav2vec2 from opt.wav2vec_model_path
    # - emotion_encoder will init its model from opt.audio2emotion_path
    # - Then agent.load_weight will load float.pth weights into agent.G (the FLOAT instance)
    #   This means float.pth must contain weights for motion_autoencoder, fmt,
    #   and also for audio_encoder.audio_projection if its not in wav2vec2-base-960h.
    agent = InferenceAgent(opt)  # This loads everything based on current separate files
    agent.G.to(opt.rank)  # Ensure everything is on CPU

    print("Collecting state dictionary from the fully loaded agent.G (FLOAT model)...")
    # agent.G is the instance of the FLOAT class.
    # Its state_dict() will contain all parameters with correct hierarchical names
    # (e.g., "motion_autoencoder.enc.net_app.convs.0.0.weight", "audio_encoder.wav2vec2.encoder.layers.0...", etc.)
    combined_state_dict = agent.G.state_dict()

    num_keys = len(combined_state_dict)
    print(f"Collected {num_keys} keys in the combined state dictionary.")
    if num_keys == 0:
        print("Error: No keys found in the state dictionary. Model loading might have failed silently.")
        return

    print(f"Saving combined state dictionary to {output_safetensors_path}...")
    save_file(combined_state_dict, output_safetensors_path)
    print(f"Successfully saved combined model to {output_safetensors_path}")
    print("Next steps:")
    print(f"1. Move '{output_safetensors_path}' to your 'ComfyUI/models/float/' directory.")
    print(f"2. Ensure 'config_wav2vec2_base.json', 'config_emotion_ser.json', and 'preprocessor_config_wav2vec2_base.json' "
          f"are in your node's '{os.path.join(base_dir, 'model_configs')}/' directory.")
    print("3. Update your node code (nodes.py, generate.py, etc.) to use these bundled files.")


if __name__ == "__main__":
    # Define where the output safetensors file will be saved by this script
    output_file = "FLOAT.safetensors"
    create_and_save_combined_model(output_file)

    # Optional: Clean up the temporary download directory
    # import shutil
    # script_base_dir = os.path.dirname(os.path.abspath(__file__))
    # temp_original_float_models_dir = os.path.join(script_base_dir, "temp_original_models")
    # if os.path.exists(temp_original_float_models_dir):
    #     print(f"Cleaning up temporary directory: {temp_original_float_models_dir}")
    #     shutil.rmtree(temp_original_float_models_dir)

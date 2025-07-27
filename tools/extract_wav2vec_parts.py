#!/usr/bin/env python3
# Gemini 2.5 Pro tool
import argparse
import os
from safetensors.torch import load_file, save_file
from collections import OrderedDict


def extract_and_save_submodel(main_model_path, output_safetensors_path, model_prefix: str):
    """
    Extracts a sub-model's weights from a main FLOAT model checkpoint.

    Args:
        main_model_path (str): Path to the unified FLOAT.safetensors file.
        output_safetensors_path (str): Path to save the extracted sub-model weights.
        model_prefix (str): The prefix for the sub-model in the main state_dict
                            (e.g., "audio_encoder.wav2vec2" or "emotion_encoder.wav2vec2_for_emotion").
    """
    if not model_prefix.endswith('.'):
        model_prefix += '.'

    print(f"Loading main model from: {main_model_path}")
    try:
        main_sd = load_file(main_model_path, device="cpu")
    except Exception as e:
        print(f"Error loading main model state_dict: {e}")
        return

    extracted_sd = OrderedDict()
    found_weights = False

    print(f"\nAttempting to extract weights with prefix '{model_prefix}':")
    for key, value in main_sd.items():
        if key.startswith(model_prefix):
            new_key = key[len(model_prefix):]
            extracted_sd[new_key] = value
            print(f"  Mapped: '{key}' -> '{new_key}'")
            found_weights = True

    if not found_weights:
        print(f"\nError: No weights found with the prefix '{model_prefix}'.")
        print("Please verify the prefix and that the unified model contains these weights.")
        return

    print(f"\nExtracted sub-model state_dict contains {len(extracted_sd)} tensors.")
    try:
        output_dir = os.path.dirname(output_safetensors_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # Note: The emotion model was originally .bin, but saving as .safetensors is better.
        # The loader node will need to know to load .safetensors.
        save_file(extracted_sd, output_safetensors_path)
        print(f"Successfully saved extracted weights to: {output_safetensors_path}")
    except Exception as e:
        print(f"Error saving extracted weights: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract a Wav2Vec2 sub-model from a unified FLOAT checkpoint.")
    parser.add_argument("main_model_path", type=str,
                        help="Path to the unified FLOAT.safetensors model file.")
    parser.add_argument("output_path", type=str,
                        help="Path to save the extracted sub-model weights (as .safetensors).")
    parser.add_argument("prefix", type=str,
                        help="Prefix of the sub-model to extract (e.g., 'audio_encoder.wav2vec2').")

    args = parser.parse_args()
    extract_and_save_submodel(args.main_model_path, args.output_path, args.prefix)


if __name__ == "__main__":
    main()

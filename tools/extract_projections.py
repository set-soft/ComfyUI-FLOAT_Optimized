#!/usr/bin/python3
import torch
import argparse
import os
from safetensors.torch import load_file, save_file
from collections import OrderedDict

# Define the expected architecture of the projection layer for key mapping
# This should match what LoadAudioProjectionLayer will create.
# nn.Sequential(
#     nn.Linear(input_dim, output_dim), # Key prefix "0"
#     nn.LayerNorm(output_dim),         # Key prefix "1"
#     nn.SiLU()                         # No learnable parameters
# )


def extract_and_save_projection_weights(main_model_path, output_safetensors_path,
                                        model_prefix="audio_encoder.audio_projection"):  # MODIFIED DEFAULT PREFIX
    """
    Extracts audio projection weights from a main FLOAT model checkpoint
    and saves them to a new .safetensors file.

    Args:
        main_model_path (str): Path to the main .safetensors or .pth FLOAT model file.
        output_safetensors_path (str): Path to save the extracted projection weights.
        model_prefix (str): The prefix for the audio_projection module in the main model's
                            state_dict.
    """
    print(f"Loading main model from: {main_model_path}")
    try:
        if main_model_path.endswith(".safetensors"):
            main_sd = load_file(main_model_path, device="cpu")
        elif main_model_path.endswith(".pth"):
            main_sd = torch.load(main_model_path, map_location="cpu")
            # Handle potential nesting in .pth files (e.g., 'state_dict', 'model')
            # Common patterns:
            if 'state_dict' in main_sd:
                main_sd = main_sd['state_dict']
            elif 'model' in main_sd and isinstance(main_sd['model'], OrderedDict):  # Often from PyTorch Lightning or similar
                main_sd = main_sd['model']
            # If the .pth is a direct save of the model's state_dict(), main_sd is already correct.
        else:
            raise ValueError("Unsupported model file type. Please use .safetensors or .pth.")
    except Exception as e:
        print(f"Error loading main model state_dict: {e}")
        return

    projection_sd = OrderedDict()
    found_weights = False

    expected_relative_keys = {
        "0.weight": f"{model_prefix}.0.weight",
        "0.bias": f"{model_prefix}.0.bias",
        "1.weight": f"{model_prefix}.1.weight",
        "1.bias": f"{model_prefix}.1.bias",
    }

    print(f"\nAttempting to extract weights with prefix '{model_prefix}':")
    for new_key, original_key in expected_relative_keys.items():
        if original_key in main_sd:
            projection_sd[new_key] = main_sd[original_key]
            print(f"  Found and mapped: '{original_key}' -> '{new_key}' (shape: {main_sd[original_key].shape})")
            found_weights = True
        else:
            print(f"  MISSING expected key in main model: '{original_key}'")

    if not found_weights:
        print(f"\nNo audio projection weights found with the prefix '{model_prefix}'.")
        print("Please verify the prefix and the model structure. The prefix should be relative to the root of the state_dict.")
        print("\nAvailable top-level keys in the loaded state_dict (first 20):")
        for i, key in enumerate(list(main_sd.keys())[:20]):
            print(f"  {key}")
        if len(main_sd.keys()) > 20:
            print(f"  ... and {len(main_sd.keys()) - 20} more keys.")
        return

    print(f"\nExtracted projection state_dict contains {len(projection_sd)} tensors.")

    try:
        # Ensure output directory exists (safer if output_path includes directories)
        output_dir = os.path.dirname(output_safetensors_path)
        if output_dir:  # Only try to make dirs if a path is specified
            os.makedirs(output_dir, exist_ok=True)

        save_file(projection_sd, output_safetensors_path)
        print(f"Successfully saved extracted audio projection weights to: {output_safetensors_path}")
    except Exception as e:
        print(f"Error saving projection weights: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract Audio Projection Layer weights from a FLOAT model checkpoint.")
    parser.add_argument("main_model_path", type=str,
                        help="Path to the main FLOAT model file (.safetensors or .pth).")
    parser.add_argument("output_path", type=str,
                        help="Path to save the extracted .safetensors projection weights.")
    # MODIFIED DEFAULT PREFIX HERE
    parser.add_argument("--prefix", type=str, default="audio_encoder.audio_projection",
                        help="Prefix for the audio projection module in the main model's state_dict. "
                        "Default: 'audio_encoder.audio_projection'")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir) and not os.path.isfile(output_dir):  # Check if it's not a file
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: Could not create output directory {output_dir}: {e}")
            return  # Exit if cannot create output directory

    if not args.output_path.endswith(".safetensors"):
        print("Warning: Output path does not end with .safetensors. It will be saved in this format.")

    extract_and_save_projection_weights(args.main_model_path, args.output_path, args.prefix)


if __name__ == "__main__":
    main()

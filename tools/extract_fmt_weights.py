#!/usr/bin/env python3
# Gemini 2.5 Pro tool
import torch
import argparse
import os
from safetensors.torch import load_file, save_file
from collections import OrderedDict


def extract_and_save_fmt_weights(main_model_path, output_safetensors_path, model_prefix="fmt."):
    """
    Extracts FlowMatchingTransformer (FMT) weights from a main FLOAT model checkpoint
    and saves them to a new .safetensors file.

    Args:
        main_model_path (str): Path to the main .safetensors or .pth FLOAT model file.
        output_safetensors_path (str): Path to save the extracted FMT weights.
        model_prefix (str): The prefix for the FMT module in the main model's state_dict.
                            Default assumes the FLOAT instance has an attribute 'fmt'.
    """
    print(f"Loading main model from: {main_model_path}")
    try:
        if main_model_path.endswith(".safetensors"):
            main_sd = load_file(main_model_path, device="cpu")
        elif main_model_path.endswith(".pth"):
            main_sd = torch.load(main_model_path, map_location="cpu")
            # Handle potential nesting in .pth files
            if 'state_dict' in main_sd:
                main_sd = main_sd['state_dict']
            elif 'model' in main_sd and isinstance(main_sd['model'], OrderedDict):
                main_sd = main_sd['model']
            # If FLOAT class instance was saved directly, main_sd is correct.
        else:
            raise ValueError("Unsupported model file type. Please use .safetensors or .pth.")
    except Exception as e:
        print(f"Error loading main model state_dict: {e}")
        return

    fmt_sd = OrderedDict()
    found_weights = False

    # Ensure the prefix ends with a dot if it's not empty, for proper key matching.
    if model_prefix and not model_prefix.endswith('.'):
        prefix_to_use = model_prefix + '.'
    else:
        prefix_to_use = model_prefix  # Handles empty prefix or already dot-terminated

    print(f"\nAttempting to extract FMT weights with prefix '{prefix_to_use}':")
    for key, value in main_sd.items():
        if key.startswith(prefix_to_use):
            # Strip the prefix to get keys relative to the FMT module
            new_key = key[len(prefix_to_use):]
            fmt_sd[new_key] = value
            print(f"  Found and mapped: '{key}' -> '{new_key}' (shape: {value.shape})")
            found_weights = True

    if not found_weights:
        print(f"\nNo FMT weights found with the prefix '{prefix_to_use}'.")
        print("Please verify the prefix and the model structure.")
        print("The prefix should point to the FlowMatchingTransformer attribute within the saved FLOAT model.")
        print("\nAvailable top-level keys in the loaded state_dict (first 20 to help debug):")
        for i, k_top in enumerate(list(main_sd.keys())[:20]):
            print(f"  {k_top}")
        if len(main_sd.keys()) > 20:
            print(f"  ... and {len(main_sd.keys()) - 20} more keys.")
        return

    print(f"\nExtracted FMT state_dict contains {len(fmt_sd)} tensors.")

    try:
        output_dir = os.path.dirname(output_safetensors_path)
        if output_dir:  # Only create if path implies directories
            os.makedirs(output_dir, exist_ok=True)
        save_file(fmt_sd, output_safetensors_path)
        print(f"Successfully saved extracted FMT weights to: {output_safetensors_path}")
    except Exception as e:
        print(f"Error saving extracted FMT weights: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract FlowMatchingTransformer (FMT) weights from a FLOAT model "
                                     "checkpoint.")
    parser.add_argument("main_model_path", type=str,
                        help="Path to the main FLOAT model file (.safetensors or .pth).")
    parser.add_argument("output_path", type=str,
                        help="Path to save the extracted .safetensors FMT weights.")
    parser.add_argument("--prefix", type=str, default="fmt",  # Default prefix for 'self.fmt'
                        help="Prefix for the FMT module in the main model's state_dict. "
                        "Default: 'fmt'")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir) and not os.path.isfile(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: Could not create output directory {output_dir}: {e}")
            return

    if not args.output_path.endswith(".safetensors"):
        print("Warning: Output path does not end with .safetensors. It will be saved in this format.")

    # The function expects prefix to end with a dot for stripping, but arg is cleaner without it
    prefix_for_extraction = args.prefix
    if prefix_for_extraction and not prefix_for_extraction.endswith('.'):
        prefix_for_extraction += '.'

    extract_and_save_fmt_weights(args.main_model_path, args.output_path, prefix_for_extraction)


if __name__ == "__main__":
    main()

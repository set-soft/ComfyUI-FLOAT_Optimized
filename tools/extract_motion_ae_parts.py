#!/usr/bin/env python3
# Gemini 2.5 Pro tool
import torch
import argparse
import os
from safetensors.torch import load_file, save_file
from collections import OrderedDict


def extract_and_save_motion_ae_part(main_model_path, output_safetensors_path, part_to_extract="enc"):
    """
    Extracts motion_autoencoder.enc or motion_autoencoder.dec weights
    from a main FLOAT model checkpoint and saves them.

    Args:
        main_model_path (str): Path to the main .safetensors or .pth FLOAT model file.
        output_safetensors_path (str): Path to save the extracted weights.
        part_to_extract (str): "enc" for Encoder, "dec" for Decoder(Synthesis).
    """
    if part_to_extract not in ["enc", "dec"]:
        print(f"Error: Invalid part_to_extract '{part_to_extract}'. Must be 'enc' or 'dec'.")
        exit(2)

    base_prefix = "motion_autoencoder"
    full_prefix = f"{base_prefix}.{part_to_extract}."  # Note the trailing dot

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
            # Add more specific .pth handling if your structure is known
        else:
            raise ValueError("Unsupported model file type. Please use .safetensors or .pth.")
    except Exception as e:
        print(f"Error loading main model state_dict: {e}")
        return

    extracted_part_sd = OrderedDict()
    found_weights = False

    print(f"\nAttempting to extract weights with prefix '{full_prefix}':")
    for key, value in main_sd.items():
        if key.startswith(full_prefix):
            # Strip the prefix to get keys relative to the Encoder/Decoder module
            new_key = key[len(full_prefix):]
            extracted_part_sd[new_key] = value
            print(f"  Found and mapped: '{key}' -> '{new_key}' (shape: {value.shape})")
            found_weights = True

    if not found_weights:
        print(f"\nNo weights found for the '{part_to_extract}' part with prefix '{full_prefix}'.")
        print("Please verify the model structure or if the main file contains these weights.")
        print("\nAvailable top-level keys in the loaded state_dict (first 20 to help debug if prefix is wrong):")
        # This helps if the user made a mistake with where motion_autoencoder is located
        keys_to_show = []
        count = 0
        for k_top in main_sd.keys():
            if count < 20:
                keys_to_show.append(k_top)
            else:
                # Check if any key starts with "motion_autoencoder" to guide the user
                if "motion_autoencoder" in k_top and count < 25:  # Show a few more if relevant
                    keys_to_show.append(k_top)
                elif count == 20:  # Only print ellipsis once
                    keys_to_show.append("...")
            count += 1
        for k_show in keys_to_show:
            print(f"  {k_show}")

        if len(main_sd.keys()) > len(keys_to_show) and keys_to_show[-1] != "...":
            print(f"  ... and {len(main_sd.keys()) - len(keys_to_show)} more keys.")
        return

    print(f"\nExtracted '{part_to_extract}' state_dict contains {len(extracted_part_sd)} tensors.")

    try:
        output_dir = os.path.dirname(output_safetensors_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        save_file(extracted_part_sd, output_safetensors_path)
        print(f"Successfully saved extracted '{part_to_extract}' weights to: {output_safetensors_path}")
    except Exception as e:
        print(f"Error saving extracted weights: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract Encoder or Decoder (Synthesis) weights from a FLOAT model "
                                     "checkpoint.")
    parser.add_argument("main_model_path", type=str,
                        help="Path to the main FLOAT model file (.safetensors or .pth).")
    parser.add_argument("output_path", type=str,
                        help="Path to save the extracted .safetensors weights.")
    parser.add_argument("part", choices=["enc", "dec"],
                        help="Which part to extract: 'enc' for Encoder, 'dec' for Decoder (Synthesis).")
    # The prefix for motion_autoencoder itself is assumed to be at the root of the state_dict.
    # If motion_autoencoder was nested (e.g., "G.motion_autoencoder"), the user would modify
    # the base_prefix in the script or we'd add another argument. For now, keeping it simple.

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir) and not os.path.isfile(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: Could not create output directory {output_dir}: {e}")
            exit(2)

    if not args.output_path.endswith(".safetensors"):
        print("Warning: Output path does not end with .safetensors. It will be saved in this format.")

    extract_and_save_motion_ae_part(args.main_model_path, args.output_path, args.part)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
# Original code from Gemini 2.5 Pro
import logging
import os
import requests
import shutil
import subprocess
import sys
from tqdm import tqdm

# ComfyUI imports
import comfy.utils
import folder_paths

from .misc import NODES_NAME

# Assuming a logger is set up
logger = logging.getLogger(f"{NODES_NAME}.downloader")
BASE_FLOAT = "https://huggingface.co/set-soft/float_advanced/resolve/main/"
# URLs for direct download of individual parts
MODEL_PART_URLS = {
    "encoder": BASE_FLOAT + "motion_autoencoder/encoder.safetensors",
    "decoder": BASE_FLOAT + "motion_autoencoder/decoder.safetensors",
    "projection": BASE_FLOAT + "audio_projections/projection.safetensors",
    "fmt": BASE_FLOAT + "fmt/fmt.safetensors?download=true",
    "wav2vec2_base": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/model.safetensors",
    # The original emotion S.E.R. model is pytorch_model.bin.
    # We will save our extracted version as .safetensors for consistency.
    "emotion_ser": "https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition/resolve/main/pytorch_model.bin"
}
# Prefixes for extraction from the unified model
EXTRACTION_PREFIXES = {
    "encoder": "motion_autoencoder.enc",
    "decoder": "motion_autoencoder.dec",
    "projection": "audio_encoder.audio_projection",
    "fmt": "fmt",
    "wav2vec2_base": "audio_encoder.wav2vec2",
    "emotion_ser": "emotion_encoder.wav2vec2_for_emotion"
}


def download_model(url: str, save_dir: str, file_name: str):
    """
    Downloads a file from a URL with progress bars for both console and ComfyUI.

    Args:
        url (str): The direct download URL for the file.
        save_dir (str): The directory where the file will be saved.
        file_name (str): The name of the file to be saved on disk.
    """
    full_path = os.path.join(save_dir, file_name)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Downloading model: {file_name}")
    logger.info(f"Source URL: {url}")
    logger.info(f"Destination: {full_path}")

    try:
        # Use a streaming request to handle large files and get content length
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Get total file size from headers
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            # --- Setup Progress Bars ---
            # Console progress bar using tqdm
            progress_bar_console = tqdm(
                total=total_size_in_bytes,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {file_name}"
            )

            # ComfyUI progress bar
            progress_bar_ui = comfy.utils.ProgressBar(total_size_in_bytes)

            # --- Download Loop ---
            downloaded_size = 0
            with open(full_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive new chunks
                        chunk_size = len(chunk)

                        # Update console progress bar
                        progress_bar_console.update(chunk_size)

                        # Update ComfyUI progress bar
                        downloaded_size += chunk_size
                        progress_bar_ui.update(chunk_size)  # ProgressBar takes absolute value, but update is incremental

                        # Write chunk to file
                        f.write(chunk)

            # --- Cleanup ---
            progress_bar_console.close()

            # Final check to see if download was complete
            if total_size_in_bytes != 0 and progress_bar_console.n != total_size_in_bytes:
                logger.error("Download failed: Size mismatch.")
                # Optional: remove partial file
                # os.remove(full_path)
                raise IOError(f"Download failed for {file_name}. Expected {total_size_in_bytes} but got "
                              f"{progress_bar_console.n}")

        logger.info(f"Successfully downloaded {file_name}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while downloading {file_name}: {e}")
        # Clean up partial file if it exists
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except OSError:
                pass
        raise
    except Exception as e:
        logger.error(f"An error occurred during download: {e}")
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except OSError:
                pass
        raise


def ensure_model_part_exists(part_key: str, sub_dir: str, file_name: str, unified_model_path: str, node_root_path: str):
    """
    Ensures a model part exists. If not, it tries to extract it from the unified
    model. If the unified model doesn't exist, it downloads the part directly.

    Args:
        part_key (str): A key from MODEL_PART_URLS/EXTRACTION_PREFIXES (e.g., "encoder").
        sub_dir (str): The subdirectory within `models/` where the part should be (e.g., "float/motion_autoencoder").
        file_name (str): The filename of the model part (e.g., "encoder.safetensors").
        unified_model_path (str): The full path to the main FLOAT.safetensors file.
        node_root_path (str): The root path of the custom node package, to find tool scripts.
    """
    part_full_path = os.path.join(folder_paths.models_dir, sub_dir, file_name)

    if os.path.exists(part_full_path):
        logger.debug(f"Model part '{file_name}' already exists. Skipping.")
        return part_full_path

    logger.warning(f"Model part not found at: {part_full_path}")

    # Step 1: Try to extract from the unified model if it exists
    if os.path.exists(unified_model_path):
        logger.info(f"Unified model found at '{unified_model_path}'. Attempting to extract '{part_key}' part.")

        # Determine which extraction script to use
        if part_key in ["encoder", "decoder"]:
            script_path = os.path.join(node_root_path, "tools", "extract_motion_ae_parts.py")
            cmd = [sys.executable, script_path, unified_model_path, part_full_path, part_key[:3]]
        elif part_key in EXTRACTION_PREFIXES:
            script_path = os.path.join(node_root_path, "tools", "extract_wav2vec_parts.py")  # A more generic script
            prefix = EXTRACTION_PREFIXES[part_key]
            cmd = [sys.executable, script_path, unified_model_path, part_full_path, prefix]
        else:
            logger.error(f"Unknown part_key '{part_key}' for extraction.")
            return None

        try:
            target_dir = os.path.dirname(part_full_path)
            # We must create the destination folder before calling the tool
            os.makedirs(target_dir, exist_ok=True)
            # Run the extraction script as a subprocess
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully extracted '{file_name}'.\n{result.stdout}")
            extra_files_path = os.path.join(node_root_path, "model_configs", part_key)
            if os.path.isdir(extra_files_path):
                shutil.copytree(extra_files_path, target_dir, dirs_exist_ok=True)
            return part_full_path  # Success!
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract '{file_name}' from unified model. Error:\n{e.stderr}")
            # Fall through to download if extraction fails.
        except FileNotFoundError:
            logger.error(f"Extraction script not found at '{script_path}'. Cannot extract. Please ensure tools are in place.")
            # Fall through to download.

    # Step 2: If extraction failed or unified model is not present, download directly
    logger.info(f"Could not extract part. Attempting to download '{file_name}' directly.")
    part_url = MODEL_PART_URLS.get(part_key)
    if not part_url:
        logger.error(f"No direct download URL configured for part_key '{part_key}'. Cannot download.")
        raise FileNotFoundError(f"Could not find or download required model part: {file_name}")
    part_url += "?download=true"

    try:
        # Use the existing downloader with UI progress bar
        download_model(url=part_url, save_dir=os.path.dirname(part_full_path), file_name=file_name)
    except Exception as e:
        logger.error(f"Direct download of '{file_name}' failed: {e}")
        raise  # Re-raise to stop execution and show error in UI.
    return part_full_path


def look_for_models(f_dir: str, default_name: str):
    models_path = os.path.join(folder_paths.models_dir, f_dir)
    if not os.path.isdir(models_path):
        try:
            os.makedirs(models_path, exist_ok=True)
        except OSError:
            pass

    if os.path.isdir(models_path):
        other_files = sorted([f for f in os.listdir(models_path) if f.endswith(".safetensors") and f != default_name])
    else:
        other_files = []

    return [default_name] + other_files


def look_for_model_dirs(f_dir: str, default_name: str):
    models_path = os.path.join(folder_paths.models_dir, f_dir)
    if not os.path.isdir(models_path):
        # Attempt to create it, but don't fail if it doesn't exist yet,
        # as the list might just show the default to download
        try:
            os.makedirs(models_path, exist_ok=True)
        except OSError:
            pass  # Oh well, list will be empty or show error.

    other_folders = []
    if os.path.isdir(models_path):
        for f_name in os.listdir(models_path):
            if f_name == default_name:
                continue
            full_path = os.path.join(models_path, f_name)
            if os.path.isdir(full_path):
                # Check for common Hugging Face model files
                if os.path.exists(os.path.join(full_path, "config.json")) and \
                    (os.path.exists(os.path.join(full_path, "pytorch_model.bin")) or
                     os.path.exists(os.path.join(full_path, "model.safetensors")) or
                     os.path.exists(os.path.join(full_path, "tf_model.h5"))):  # Added safetensors and tf
                    other_folders.append(f_name)

    return [default_name] + sorted(other_folders)

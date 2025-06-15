# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
# Original code from Gemini 2.5 Pro
import logging
import os
import requests
from tqdm import tqdm

# ComfyUI imports
import comfy.utils

from .misc import NODES_NAME

# Assuming a logger is set up
logger = logging.getLogger(f"{NODES_NAME}.downloader")


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

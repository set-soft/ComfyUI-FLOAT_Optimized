# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
#
# Audio resampler and mono conversion using librosa
# Gemini 2.5 Pro code
import librosa
import logging
import numpy as np
import torch
from . import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.resample")


def resample_audio_numpy(audio_data_numpy: np.ndarray, original_sr: int, target_sr: int,
                         res_type: str = 'soxr_hq'  # High-quality resampler by default
                         ) -> np.ndarray:
    """
    Resamples a NumPy audio array from an original sample rate to a target sample rate.

    Args:
        audio_data_numpy (np.ndarray): The input audio data.
            Expected shape: (num_channels, num_samples) or (num_samples,) for mono.
        original_sr (int): The original sample rate of the audio data.
        target_sr (int): The desired target sample rate.
        res_type (str): The resampling type to use.
            Common options: 'soxr_hq' (high quality), 'soxr_mq' (medium quality),
                            'soxr_lq' (low quality), 'kaiser_best', 'kaiser_fast',
                            'scipy', 'polyphase'.
                            'soxr_**' usually require the 'soxr' package to be installed.

    Returns:
        np.ndarray: The resampled audio data with the same number of channels as input.
    """
    if original_sr == target_sr:
        logger.debug(f"Original sample rate ({original_sr} Hz) is already the target sample rate ({target_sr} Hz)."
                     " No resampling needed.")
        return audio_data_numpy

    logger.debug(f"Resampling from {original_sr} Hz to {target_sr} Hz using '{res_type}'...")

    # librosa.resample works best with floating point audio between -1 and 1.
    # If your NumPy array is integer type (e.g., int16 from a WAV file),
    # you might want to convert it to float first if it's not already.
    # Assuming audio_data_numpy is already float. If not, you'd do something like:
    # if audio_data_numpy.dtype != np.float32 and audio_data_numpy.dtype != np.float64:
    #     if np.issubdtype(audio_data_numpy.dtype, np.integer):
    #         audio_data_numpy = audio_data_numpy.astype(np.float32) / np.iinfo(audio_data_numpy.dtype).max
    #     else:
    #         # Or raise an error if unexpected dtype
    #         audio_data_numpy = audio_data_numpy.astype(np.float32)

    # librosa.resample expects y to be shape (num_samples,) or (2, num_samples) for stereo.
    # If your input is (batch_size, num_channels, num_samples), you'll need to loop
    # or adapt. For typical audio processing outside batch contexts, it's (channels, samples).

    if audio_data_numpy.ndim == 1:  # Mono audio (samples,)
        resampled_audio = librosa.resample(y=audio_data_numpy, orig_sr=original_sr, target_sr=target_sr, res_type=res_type)
    elif audio_data_numpy.ndim == 2:  # Potentially (channels, samples)
        # Check if it's (channels, samples) or (samples, channels)
        # Assuming (channels, samples) which is more common for librosa processing internally.
        # If it were (samples, channels), you'd transpose: audio_data_numpy.T

        num_channels = audio_data_numpy.shape[0]
        # Heuristic: if many "channels" and few "samples", it might be transposed
        if num_channels > audio_data_numpy.shape[1] and num_channels > 16:
            logger.warning(f"Warning: Input shape {audio_data_numpy.shape} might be (samples, channels) instead of "
                           "(channels, samples). Transposing for resampling.")
            audio_data_numpy_transposed = audio_data_numpy.T
            resampled_channels = []
            for ch_idx in range(audio_data_numpy_transposed.shape[0]):  # Iterate over transposed channels
                channel_data = audio_data_numpy_transposed[ch_idx, :]
                resampled_channel = librosa.resample(y=channel_data, orig_sr=original_sr, target_sr=target_sr,
                                                     res_type=res_type)
                resampled_channels.append(resampled_channel)
            resampled_audio = np.stack(resampled_channels, axis=0).T  # Stack and transpose back to (samples, channels)
        else:  # Assume (channels, samples)
            resampled_channels = []
            for ch_idx in range(num_channels):
                channel_data = audio_data_numpy[ch_idx, :]
                resampled_channel = librosa.resample(y=channel_data, orig_sr=original_sr, target_sr=target_sr,
                                                     res_type=res_type)
                resampled_channels.append(resampled_channel)
            resampled_audio = np.stack(resampled_channels, axis=0)  # Output will be (channels, samples)
    else:
        raise ValueError(f"Unsupported audio_data_numpy ndim: {audio_data_numpy.ndim}. Expected 1 or 2.")

    logger.debug(f"Resampling complete. Original shape: {audio_data_numpy.shape}, New shape: {resampled_audio.shape}")
    return resampled_audio


def comfy_audio_to_librosa_mono(comfy_audio_tensor: torch.Tensor, cur_sr: int, target_sr: int) -> np.ndarray:
    """
    Converts a ComfyUI audio tensor to a mono NumPy array suitable for Librosa.

    Args:
        comfy_audio_tensor (torch.Tensor):
            The input audio tensor from ComfyUI.
            Expected shape: (batch_size, channels, num_samples).
            Typically batch_size is 1.

    Returns:
        np.ndarray:
            A mono audio waveform as a NumPy array with shape (num_samples,).
    """
    if not isinstance(comfy_audio_tensor, torch.Tensor):
        raise TypeError(f"Input must be a PyTorch Tensor, got {type(comfy_audio_tensor)}")

    if comfy_audio_tensor.ndim != 3:
        raise ValueError(
            f"Expected a 3D tensor (batch_size, channels, num_samples), "
            f"but got {comfy_audio_tensor.ndim}D tensor with shape {comfy_audio_tensor.shape}")

    # 1. Handle batch dimension: Assume we process the first item if batch_size > 1
    #    or just squeeze if batch_size is indeed 1.
    if comfy_audio_tensor.shape[0] > 1:
        logger.warning(f"Warning: Input tensor has batch_size {comfy_audio_tensor.shape[0]}. "
                       "Processing only the first audio in the batch.")
        audio_tensor = comfy_audio_tensor[0]  # Shape: (channels, num_samples)
    else:
        audio_tensor = comfy_audio_tensor.squeeze(0)  # Shape: (channels, num_samples)

    # 2. Move to CPU (if it's on GPU) and convert to NumPy
    #    Librosa operates on NumPy arrays on the CPU.
    waveform_np = audio_tensor.cpu().numpy()  # Shape: (channels, num_samples)

    # 3. Convert to mono if it's not already
    #    waveform_np has shape (channels, num_samples)
    #    librosa.to_mono expects a 2D array (channels, n_samples) or 1D (n_samples)
    #    If it's already (1, num_samples), librosa.to_mono will correctly make it (num_samples,)
    if waveform_np.ndim == 1:  # Already mono (e.g. if squeeze(0) made it 1D from (1,1,N))
        mono_waveform_np = waveform_np
    elif waveform_np.shape[0] == 1:  # Explicitly mono with channel dimension
        mono_waveform_np = waveform_np[0, :]
    elif waveform_np.shape[0] > 1:  # Stereo or multi-channel
        mono_waveform_np = librosa.to_mono(waveform_np)
    else:  # Should not happen if input validation is correct
        raise ValueError(f"Unexpected waveform shape after processing: {waveform_np.shape}")

    # Ensure it's float32, which librosa typically uses, though to_mono usually preserves float type.
    if mono_waveform_np.dtype != np.float32:
        mono_waveform_np = mono_waveform_np.astype(np.float32)

    # Make the sample rate match the model training data, Wav2Vec needs 16k
    if cur_sr != target_sr:
        mono_waveform_np = resample_audio_numpy(mono_waveform_np, cur_sr, target_sr)

    return mono_waveform_np


# --- Example Usage ---
if __name__ == '__main__':
    # 1. Create some dummy audio data
    original_sample_rate = 22050
    duration_seconds = 3
    num_samples_original = original_sample_rate * duration_seconds

    # Mono example
    logger.info("--- Mono Resampling Example ---")
    # Create a sine wave as mono audio
    frequency_mono = 440  # A4 note
    t_mono = np.linspace(0, duration_seconds, num_samples_original, endpoint=False)
    mono_audio = 0.5 * np.sin(2 * np.pi * frequency_mono * t_mono)
    logger.info(f"Original mono audio shape: {mono_audio.shape}, SR: {original_sample_rate} Hz")

    # Resample mono audio to 44100 Hz
    target_sr_44100 = 44100
    resampled_mono_44100 = resample_audio_numpy(mono_audio, original_sample_rate, target_sr_44100)
    # Expected number of samples: num_samples_original * (target_sr / original_sr)
    expected_samples_44100 = int(num_samples_original * (target_sr_44100 / original_sample_rate))
    logger.info(f"Resampled mono to {target_sr_44100} Hz, shape: {resampled_mono_44100.shape} "
                f"(expected ~{expected_samples_44100} samples)")
    assert abs(resampled_mono_44100.shape[0] - expected_samples_44100) < 2  # Allow for small rounding in length

    # Resample mono audio back to original SR (e.g., 22050 Hz from 44100 Hz)
    resampled_mono_back = resample_audio_numpy(resampled_mono_44100, target_sr_44100, original_sample_rate)
    logger.info(f"Resampled mono back to {original_sample_rate} Hz, shape: {resampled_mono_back.shape}"
                f" (expected {num_samples_original} samples)")
    assert abs(resampled_mono_back.shape[0] - num_samples_original) < 2

    # Stereo example
    logger.info("\n--- Stereo Resampling Example ---")
    # Create two sine waves for stereo
    frequency_left = 440
    frequency_right = 660
    t_stereo = np.linspace(0, duration_seconds, num_samples_original, endpoint=False)
    left_channel = 0.5 * np.sin(2 * np.pi * frequency_left * t_stereo)
    right_channel = 0.3 * np.sin(2 * np.pi * frequency_right * t_stereo)
    stereo_audio = np.stack((left_channel, right_channel), axis=0)  # Shape: (2, num_samples_original)
    logger.info(f"Original stereo audio shape: {stereo_audio.shape}, SR: {original_sample_rate} Hz")

    # Resample stereo audio to 48000 Hz
    target_sr_48000 = 48000
    resampled_stereo_48000 = resample_audio_numpy(stereo_audio, original_sample_rate, target_sr_48000)
    expected_samples_48000 = int(num_samples_original * (target_sr_48000 / original_sample_rate))
    logger.info(f"Resampled stereo to {target_sr_48000} Hz, shape: {resampled_stereo_48000.shape}"
                f" (expected (2, ~{expected_samples_48000}) samples)")
    assert resampled_stereo_48000.shape[0] == 2
    assert abs(resampled_stereo_48000.shape[1] - expected_samples_48000) < 2

    # Resample stereo audio from 48000 Hz back to 44100 Hz
    # This is a more common downsampling then up/down to original.
    target_sr_again_44100 = 44100
    resampled_stereo_again = resample_audio_numpy(resampled_stereo_48000, target_sr_48000, target_sr_again_44100)
    expected_samples_again = int(resampled_stereo_48000.shape[1] * (target_sr_again_44100 / target_sr_48000))
    logger.info(f"Resampled stereo from {target_sr_48000} Hz to {target_sr_again_44100} Hz, "
                f"shape: {resampled_stereo_again.shape} (expected (2, ~{expected_samples_again}) samples)")
    assert resampled_stereo_again.shape[0] == 2
    assert abs(resampled_stereo_again.shape[1] - expected_samples_again) < 2

    # Test with target SR same as original
    logger.info("\n--- Test No Resampling Needed ---")
    no_resample_needed = resample_audio_numpy(mono_audio, original_sample_rate, original_sample_rate)
    assert np.array_equal(mono_audio, no_resample_needed)

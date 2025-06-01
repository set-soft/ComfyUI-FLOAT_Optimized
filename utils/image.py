# -*- coding: utf-8 -*-
# Copyright (c) 2025 DeepBrain AI Research
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
# Mostly from Gemini 2.5 Pro, the face align code is from DeepBrain AI Research
import cv2
import face_alignment
import logging
import numpy as np
import torch

logger = logging.getLogger("FLOAT_Optimized.image")
fa_instance = None


def get_face_align():
    global fa_instance
    if fa_instance is None:
        fa_instance = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    return fa_instance


def hex_to_rgb_uint8(hex_color: str) -> tuple[int, int, int]:
    """ Converts a hex color string (e.g., #RRGGBB) to an (R, G, B) tuple of uint8. """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        logger.warning(f"Invalid hex color string: '{hex_color}'. Defaulting to black.")
        return (0, 0, 0)
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        logger.warning(f"Invalid characters in hex color string: '{hex_color}'. Defaulting to black.")
        return (0, 0, 0)


def convert_rgba_to_rgb_numpy(image_rgba_uint8: np.ndarray,  # Expected (H, W, 4), uint8, [0-255]
                              strategy: str,
                              background_color_hex: str) -> np.ndarray:  # Returns (H, W, 3), uint8, [0-255]
    """
    Converts an RGBA NumPy array to RGB using the specified strategy.
    """
    logger.debug(f"Converting RGBA to RGB using strategy: {strategy}")

    background_color_rgb_uint8 = hex_to_rgb_uint8(background_color_hex)

    rgb_image = image_rgba_uint8[..., :3]     # Extract RGB channels
    alpha_channel = image_rgba_uint8[..., 3]  # Extract Alpha channel

    if strategy == "discard_alpha":
        return rgb_image

    elif strategy == "blend_with_color":
        # Alpha blending: output = foreground * alpha + background * (1 - alpha)
        # Normalize alpha to [0, 1] for blending calculation
        alpha_float = alpha_channel.astype(np.float32) / 255.0
        alpha_expanded = alpha_float[..., np.newaxis]  # Make it (H, W, 1) for broadcasting

        bg_color_array = np.array(background_color_rgb_uint8, dtype=np.uint8)  # (3,)

        # Create a background image of the same size as rgb_image
        background_image = np.full_like(rgb_image, bg_color_array, dtype=np.uint8)

        blended_image_float = rgb_image.astype(np.float32) * alpha_expanded + \
            background_image.astype(np.float32) * (1.0 - alpha_expanded)
        return np.clip(blended_image_float, 0, 255).astype(np.uint8)

    elif strategy == "replace_with_color":
        # Identify fully transparent pixels (alpha == 0)
        # Using a small tolerance for floating point alpha if it were float, but here it's uint8
        transparent_mask = alpha_channel == 0  # Boolean mask (H, W)

        # Create a copy to modify
        output_rgb_image = np.copy(rgb_image)

        # Replace color in transparent regions
        output_rgb_image[transparent_mask] = background_color_rgb_uint8
        return output_rgb_image

    else:
        logger.warning(f"Unknown RGBA conversion strategy: '{strategy}'. Defaulting to 'discard_alpha'.")
        return rgb_image


def img_tensor_2_np_array(comfy_image_tensor: torch.Tensor, rgba_conversion: str, bkg_color_hex: str) -> np.ndarray:
    """
    Converts a ComfyUI image tensor into an internal np.ndarray

    Args:
        comfy_image_tensor (Union[str, torch.Tensor]):
              A ComfyUI image tensor, typically in
              (B, H, W, C) or (H, W, C) format, float32, range [0,1], RGB.

    Returns:
        np.ndarray: The image as a NumPy array in (H, W, C) format,
                    uint8, range [0, 255], RGB.
    """
    # Ensure tensor is on CPU
    if comfy_image_tensor.device.type != 'cpu':
        comfy_image_tensor = comfy_image_tensor.cpu()

    # Handle batch dimension and select the first image
    # ComfyUI image tensors are typically (batch, height, width, channels)
    if comfy_image_tensor.ndim == 4:
        if comfy_image_tensor.shape[0] != 1:
            logger.warning(f"Warning: Input tensor has batch_size {comfy_image_tensor.shape[0]}. "
                           "img_tensor_2_np_array is processing only the first image.")
        img_tensor_hwc = comfy_image_tensor[0]  # Shape: (H, W, C)
    elif comfy_image_tensor.ndim == 3:  # Assuming (H, W, C)
        img_tensor_hwc = comfy_image_tensor
    else:
        raise ValueError(f"Unsupported tensor ndim: {comfy_image_tensor.ndim}. "
                         "Expected 3 (H,W,C) or 4 (B,H,W,C).")

    # Convert to NumPy array
    # Tensor is assumed to be float32, range [0,1], and RGB channels
    numpy_image = img_tensor_hwc.numpy()

    # Scale from [0, 1] to [0, 255] and convert to uint8
    # Using np.clip to ensure values are strictly within [0, 255] after scaling,
    # as floating point inaccuracies might push values slightly out of [0,1].
    numpy_image_scaled_uint8 = np.clip(numpy_image * 255.0, 0, 255).astype(np.uint8)

    # Is in RGBA format?
    if numpy_image_scaled_uint8.shape[2] == 4:
        # Remove the alpha channel
        return convert_rgba_to_rgb_numpy(numpy_image_scaled_uint8, rgba_conversion, bkg_color_hex)

    # The tensor is assumed to be RGB, and the target format is RGB.
    return numpy_image_scaled_uint8


@torch.no_grad()
def process_img(img: np.ndarray, input_size: int, margin: float = 1.6) -> np.ndarray:
    mult = 360. / img.shape[0]

    resized_img = cv2.resize(img, dsize=(0, 0), fx=mult, fy=mult, interpolation=cv2.INTER_AREA
                             if mult < 1. else cv2.INTER_CUBIC)
    bboxes = get_face_align().face_detector.detect_from_image(resized_img)
    if not bboxes:
        msg = "Failed to detect any face in the image, no face align performed"
        logger.warning(msg)

        my = int(img.shape[0] / 2)
        mx = int(img.shape[1] / 2)
        bs = min(mx, my)
    else:
        bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score)
                  for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
        bboxes = bboxes[0]  # Just use first bbox

        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my = int((bboxes[1] + bboxes[3]) / 2)
        mx = int((bboxes[0] + bboxes[2]) / 2)

        bs = int(max(bsy, bsx) * margin)
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx = my + bs, mx + bs      # BBox center y, bbox center x

    crop_img = img[my - bs:my + bs, mx - bs:mx + bs]
    crop_img = cv2.resize(crop_img, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA
                          if mult < 1. else cv2.INTER_CUBIC)
    return crop_img

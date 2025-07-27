import cv2
import numpy as np
from PIL import Image


default_dilation_px = 0  


def _dilate_mask(mask_pil: Image.Image, dilation_px: int) -> Image.Image:
    if dilation_px <= 0:
        return mask_pil

    if mask_pil.mode != "L":
        mask_pil = mask_pil.convert("L")

    mask_np = np.array(mask_pil, dtype=np.uint8)
    k = dilation_px * 2 + 1
    kernel = np.ones((k, k), np.uint8)
    dilated_np = cv2.dilate(mask_np, kernel, iterations=1)

    return Image.fromarray(dilated_np, mode="L")


def overlay_masks_on_image(
    image: Image.Image,
    segmentation_masks,
    *,
    overlay_color: tuple = (255, 0, 0, 128),
    dilation_px: int = default_dilation_px,
):

    raw_image_rgba = image.convert("RGBA")
    final_image = raw_image_rgba.copy()

    if getattr(segmentation_masks, "ndim", None) == 4 and segmentation_masks.shape[1] == 1:
        segmentation_masks = segmentation_masks[:, 0, :, :]

    for mask in segmentation_masks:
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()

        if mask.ndim == 3 and mask.shape[0] == 3:
            mask = np.transpose(mask, (1, 2, 0))[:, :, 0]

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

        if mask_pil.size != raw_image_rgba.size:
            mask_pil = mask_pil.resize(raw_image_rgba.size, resample=Image.NEAREST)

        mask_pil = _dilate_mask(mask_pil, dilation_px)

        overlay = Image.new("RGBA", raw_image_rgba.size, overlay_color)
        final_image = Image.composite(overlay, final_image, mask_pil)

    return final_image.convert("RGB")


def create_black_white_mask(
    segmentation_masks,
    *,
    threshold: float = 0.5,
    combine: bool = True,
    dilation_px: int = default_dilation_px,
) -> Image.Image:

    if hasattr(segmentation_masks, "cpu"):
        seg = segmentation_masks.cpu().numpy()
    else:
        seg = segmentation_masks

    if seg.ndim == 4:
        seg = seg[:, 0, :, :]

    if seg.ndim == 3:
        binary = np.any(seg >= threshold, axis=0) if combine else np.clip(np.sum(seg >= threshold, axis=0), 0, 1)
    elif seg.ndim == 2:
        binary = seg >= threshold
    else:
        raise ValueError(f"Unexpected mask shape: {seg.shape}")

    mask_pil = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")

    mask_pil = _dilate_mask(mask_pil, dilation_px)

    return mask_pil

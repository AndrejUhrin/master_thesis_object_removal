from PIL import Image
import numpy as np
import cv2

def resize_long_side(img, target_long: int = 2048, *, nearest: bool = False):
    if isinstance(img, Image.Image):
        w, h = img.size
    elif isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    else:
        raise TypeError(f"Unsupported type: {type(img)}")

    if max(w, h) == target_long:
        return img.copy() if isinstance(img, Image.Image) else img.copy()

    scale = target_long / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)

    if isinstance(img, Image.Image):
        resample = Image.NEAREST if nearest else Image.LANCZOS
        return img.resize((new_w, new_h), resample)

    inter = cv2.INTER_NEAREST if nearest else cv2.INTER_LANCZOS4
    return cv2.resize(img, (new_w, new_h), interpolation=inter)


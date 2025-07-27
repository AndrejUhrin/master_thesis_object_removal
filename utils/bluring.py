import cv2, numpy as np

def odd(x):
    return int(x) | 1

def adaptive_blur(img_bgr: np.ndarray,
                  mask_gray: np.ndarray,
                  k_min: int = 21,
                  k_max: int = 151,
                  strength: float = 0.5) -> np.ndarray:
    out = img_bgr.copy()
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_gray, 8)

    for i in range(1, n_labels):                       
        x, y, w, h, _ = stats[i]
        comp_mask = (labels == i).astype(np.uint8) * 255

        k = odd(np.clip(strength * max(w, h), k_min, k_max))
        print(f"Object {i}: size=({w}x{h}), kernel={k}")  
        feather = odd(int(k * 0.2))                    

        soft_mask  = cv2.GaussianBlur(comp_mask, (feather, feather), 0)
        blurred_roi = cv2.GaussianBlur(img_bgr[y:y+h, x:x+w], (k, k), 0)

        roi_mask  = soft_mask[y:y+h, x:x+w].astype(np.float32) / 255.
        roi_mask3 = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)

        out[y:y+h, x:x+w] = (img_bgr[y:y+h, x:x+w] * (1 - roi_mask3) + blurred_roi * roi_mask3 ).astype(np.uint8)
    return out
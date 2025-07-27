# canny_mask_utils.py

import cv2
import numpy as np
from typing import List, Dict, Any

def compute_edge_map(
    img_bgr: np.ndarray,
    low_thresh: int = 20,
    high_thresh: int = 70
) -> np.ndarray:

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low_thresh, high_thresh)


def analyze_segmentation_edges(
    img_bgr: np.ndarray,
    mask_bool: np.ndarray,
    margin: int = 60,
    edge_threshold: float = 0.027
) -> List[Dict[str, Any]]:

    edges = compute_edge_map(img_bgr)

    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    H, W = edges.shape
    results = []

    def _border_ring(mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((2*margin + 1, 2*margin + 1), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        return dilated & ~mask

    for cnt in contours:
        comp_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(comp_mask, [cnt], -1, color=1, thickness=-1)
        comp_bool = comp_mask.astype(bool)

        ring = _border_ring(comp_bool)

        E = int(np.count_nonzero(edges[ring]))
        T = int(np.count_nonzero(ring))
        edge_ratio = float(E) / T if T else 0.0

        action = "blur" if edge_ratio > edge_threshold else "inpaint"

        results.append({
            "contour":    cnt,
            "edge_ratio": edge_ratio,
            "action":     action
        })

    return results

import csv
import json
from pathlib import Path

import cv2
import numpy as np

cov_thresh  = 0.70        

script_dir = Path(__file__).resolve().parent      
project_root = script_dir.parents[1]                
pred_base = project_root / "results_on_unseen_dataset" / "json_results_post"
gt_base = project_root / "ground_truth_dataset"
store = project_root / "validation_on_gt" / "results_final_evaluation_dataset"
csv_path = store / "detection_metrics_final_evaluation_datset.csv"


def load_gt_mask(mask_path: Path):

    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x + w, y + h))

    return boxes, mask


def extract_pred_boxes(json_path: Path, size):
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    data = json.loads(json_path.read_text())

    if "detections" in data:
        raw_boxes = [tuple(det["box"]) for det in data["detections"]]
    else:
        raw_boxes = data.get("post_filter_boxes_enlarged",
                             data.get("pre_filter_boxes_enlarged", []))
        raw_boxes = [tuple(b) for b in raw_boxes]

    for x1, y1, x2, y2 in raw_boxes:
        cv2.rectangle(mask,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      255, -1)

    return raw_boxes, mask


def pixel_cov_match(gt_box, gt_mask, pred_mask):
    x1, y1, x2, y2 = gt_box
    gt_patch   = (gt_mask[y1:y2, x1:x2] > 0)
    pred_patch = (pred_mask[y1:y2, x1:x2] > 0)
    inter = np.logical_and(gt_patch, pred_patch).sum()
    area_gt = gt_patch.sum()
    return (area_gt > 0) and ((inter / area_gt) >= cov_thresh)


rows   = []
tot_tp = tot_fp = tot_fn = 0

js_map = {p.stem: p for p in pred_base.glob("*.json")}
gt_map = {
    p.stem: p
    for p in gt_base.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
}

for stem in sorted(js_map.keys() & gt_map.keys()):
    jfile, gt_img = js_map[stem], gt_map[stem]

    img = cv2.imread(str(gt_img))
    h, w = img.shape[:2]

    pred_boxes, pred_mask = extract_pred_boxes(jfile, (w, h))
    gt_boxes, gt_mask     = load_gt_mask(gt_img)

    tp = sum(1 for gt in gt_boxes
             if pixel_cov_match(gt, gt_mask, pred_mask))
    fn = len(gt_boxes) - tp

    fp = 0
    for x1, y1, x2, y2 in pred_boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        overlap = np.logical_and(
            gt_mask[y1:y2, x1:x2] > 0,
            pred_mask[y1:y2, x1:x2] > 0
        ).any()
        if not overlap:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0

    tot_tp += tp
    tot_fp += fp
    tot_fn += fn

    rows.append({
        "image":     stem,
        "action": "object_removal",
        "TP":        tp,
        "FP":        fp,
        "FN":        fn,
        "precision": precision,
        "recall":    recall
    })

gt_tp, gt_fp, gt_fn = tot_tp, tot_fp, tot_fn

for stem in sorted(js_map.keys() - gt_map.keys()):
    data = json.loads(js_map[stem].read_text())
    if "detections" in data:
        pred_boxes = [tuple(d["box"]) for d in data["detections"]]
    else:
        pred_boxes = data.get("post_filter_boxes_enlarged",
                              data.get("pre_filter_boxes_enlarged", []))

    fp = len(pred_boxes)
    tot_fp += fp

    rows.append({
        "image": stem ,
        "action": "no_object_removal",
        "TP": 0,
        "FP": fp,
        "FN": 0,
        "precision": 0.0,
        "recall": 0.0,
    })

overall_p_gt = gt_tp / (gt_tp + gt_fp) if (gt_tp + gt_fp) else 0.0
overall_r_gt = gt_tp / (gt_tp + gt_fn) if (gt_tp + gt_fn) else 0.0


rows.append({
    "image":     "OVERALL (Needs object removal)",
    "TP":        gt_tp,
    "FP":        gt_fp,
    "FN":        gt_fn,
    "precision": overall_p_gt,
    "recall":    overall_r_gt
})

overall_p_all = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
overall_r_all = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0


rows.append({
    "image":     "OVERALL (Includes no-removal)",
    "TP":        tot_tp,
    "FP":        tot_fp,
    "FN":        tot_fn,
    "precision": overall_p_all,
    "recall":    overall_r_all
})

store.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Metrics saved to {csv_path}  (images evaluated: {len(rows)-2})")
print(f"Pixel-coverage threshold ≥ {cov_thresh}")

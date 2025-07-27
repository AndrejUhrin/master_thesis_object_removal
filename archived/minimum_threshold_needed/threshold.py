import json, csv
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

cov_thresh  = 0.70      
use_iou     = False     
script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parents[1]

pred_base = project_root / "results_pipeline_optimization_dataset" / "results_2_threshold_testing" / "json_015"
gt_base   = project_root / "ground_truth_dataset"
store     = project_root / "archived/minimum_threshold_needed"
label_csv = store / "label_score_stats.csv"

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih   = max(0, ix2-ix1), max(0, iy2-iy1)
    inter    = iw * ih
    if inter == 0: return 0.0
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / (area_a + area_b - inter)

def gt_boxes_and_mask(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(path)
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        boxes.append((x,y,x+w,y+h))
    return boxes, mask

def cov_gt_fraction(gt_box, gt_mask, pred_box):
    gx1, gy1, gx2, gy2 = map(int, gt_box)
    px1, py1, px2, py2 = map(int, pred_box)
    ix1, iy1 = max(gx1, px1), max(gy1, py1)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    gt_patch = gt_mask[gy1:gy2, gx1:gx2] > 0
    if gt_patch.sum() == 0: return 0.0
    inter_patch = gt_mask[iy1:iy2, ix1:ix2] > 0
    return inter_patch.sum() / gt_patch.sum()

def load_preds(json_path: Path):
    data = json.loads(json_path.read_text())
    return [(tuple(d["box"]), float(d.get("score", 0.0)), str(d.get("label","")))
            for d in data["detections"]]

def main():
    store.mkdir(parents=True, exist_ok=True)
    label_scores_tp = defaultdict(list)

    js_map = {p.stem: p for p in pred_base.glob("*.json")}
    gt_map = {p.stem: p for p in gt_base.iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    for stem in sorted(js_map.keys() & gt_map.keys()):
        jfile = js_map[stem]
        gt_img = gt_map[stem]

        img = cv2.imread(str(gt_img))
        if img is None:
            print(f"Skipping unreadable GT: {gt_img}")
            continue

        gt_boxes, gt_mask = gt_boxes_and_mask(gt_img)
        preds = load_preds(jfile)

        for (pb, score, label) in preds:
            best = 0.0
            if use_iou:
                for gb in gt_boxes:
                    best = max(best, iou(gb, pb))
            else:
                for gb in gt_boxes:
                    best = max(best, cov_gt_fraction(gb, gt_mask, pb))
            if best >= cov_thresh:
                label_scores_tp[label].append(score)

    if label_scores_tp:
        rows = []
        for label, scores in label_scores_tp.items():
            arr = np.array(scores, dtype=float)
            rows.append({
                "label":        label,
                "count_tp":     int(arr.size),
                "min_score_tp": float(arr.min()),
                "avg_score_tp": float(arr.mean()),
                "max_score_tp": float(arr.max())
            })
        with label_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    else:
        with label_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["label","count_tp","min_score_tp","avg_score_tp","max_score_tp"])
            writer.writeheader()

    print(f"Saved per-label stats to {label_csv}")

if __name__ == "__main__":
    main()

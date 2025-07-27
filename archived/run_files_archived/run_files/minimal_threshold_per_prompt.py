import json, shutil
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parents[2]
sys.path.insert(0, str(repo_root))

from utils.clip import load_clip, classify_image
from utils.owl import load_owl, detect_with_owl
from utils.box_mask import enlarge_box, draw_annotated

input_dir = repo_root / "pipeline_optimization_dataset"
base_path = repo_root / "archived" / "run_files_archived" / "archived_results"/ "minimum_threshold_needed"


pre_raw_ann = base_path / "pre_filter_annotated"
post_ann_out = base_path / "post_filter_annotated"
images_no_detection = base_path / "no_object_removal"
pre_json_out = base_path / "json_results_pre"
post_json_out = base_path / "json_results_post"

for d in (pre_raw_ann, post_ann_out,
          images_no_detection,
          pre_json_out, post_json_out):
    d.mkdir(parents=True, exist_ok=True)


n_images = None    
owl_thresh = 0.15
jpeg_q = 95
box_scale = 1.1

print("Loading models")
processor_clip, model_clip = load_clip()
processor_owl, model_owl = load_owl()
print("Models loaded\n")

# Labels and threshold confidence per each label
text_labels_outside = [[
    "a house number", "a license plate", "person", "a face",
    "a religious symbol", "a political symbol", "a cat", "a dog",
]]
text_labels_inside = [[
    "a calendar", "a license plate", "a paper", "person",
    "a framed picture", "a picture", "a poster board",
    "a name", "a face", "a religious symbol", "a political symbol",
    "a sex toy", "a nude image", "a cat", "a dog",
    "a mirror", "a window", "a television"
]]
per_label_thresh = {
    "a calendar": 0.20, "a paper": 0.15, "a house number": 0.15,
    "a license plate": 0.15, "person": 0.16, "a framed picture": 0.15,
    "a picture": 0.15, "a poster board": 0.15, "a name": 0.16,
    "a face": 0.15, "a religious symbol": 0.16, "a political symbol": 0.21,
    "a sex toy": 0.23, "a nude image": 0.24, "a cat": 0.17, "a dog": 0.28,
    "a mirror": 0.30, "a window": 0.30, "a television": 0.50
}
default_thresh = owl_thresh

font = ImageFont.truetype("/Library/fonts/Arial.ttf", size=50)

all_imgs = sorted([
    p for p in input_dir.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])
if n_images:
    all_imgs = all_imgs[:n_images]
print(f"Processing {len(all_imgs)} images…\n")

for idx, img_path in enumerate(all_imgs, start=1):
    stem = img_path.stem
    print(f"[{idx}/{len(all_imgs)}] {img_path.name}")

    # 1) CLIP decides if the image is indoor or outdoor scene
    image = Image.open(img_path).convert("RGB")
    w, h  = image.size
    inout = classify_image(image, processor_clip, model_clip)
    labs  = text_labels_inside if inout == "an indoor scene" else text_labels_outside

    # 2) OWLv2 object detection
    boxes_p, scores, labels = detect_with_owl(
        image, labs, processor_owl, model_owl, threshold=owl_thresh
    )
    raw_boxes = [b.tolist() for b in boxes_p]

    draw_annotated(
        image, raw_boxes, [float(s.item()) for s in scores], labels, font=font
    ).save(pre_raw_ann / f"{stem}.jpg", quality=jpeg_q)

    detections = [
        {"box": box, "label": lab, "score": float(scr.item())}
        for box, lab, scr in zip(raw_boxes, labels, scores)
    ]
    with open(pre_json_out / f"{stem}.json", "w") as f:
        json.dump(
            {"image": str(img_path.relative_to(input_dir)), "detections": detections},
            f, indent=2
        )
    # 3) filtering detections based on threshold + logic on background prompts(television, window, mirror)
    kept = [
        (b, s, l) for b, s, l in zip(raw_boxes, scores, labels)
        if s.item() >= per_label_thresh.get(l, default_thresh)
    ]
    if kept:
        boxes_f, scores_f, labels_f = map(list, zip(*kept))
    else:
        boxes_f, scores_f, labels_f = [], [], []
        print("no boxes survive score filter; skipping.\n")

    if boxes_f:
        person_or_nude = {"person", "nude image", "nude image"}
        always_drop    = {"television", "window", "mirror"}

        remove_idx = set()                 

        for i, lab in enumerate(labels_f):
            if lab in always_drop:
                remove_idx.add(i)

        for i, (box_i, lab_i) in enumerate(zip(boxes_f, labels_f)):
            if lab_i not in ("television", "window"):
                continue                     

            x0, y0, x1, y1 = box_i
            A_i = max(0, x1 - x0) * max(0, y1 - y0)
            if A_i == 0:
                continue

            for j, (box_j, lab_j) in enumerate(zip(boxes_f, labels_f)):
                if j == i:
                    continue                

                x0j, y0j, x1j, y1j = box_j
                iw = max(0, min(x1, x1j) - max(x0, x0j))
                ih = max(0, min(y1, y1j) - max(y0, y0j))
                if iw * ih == 0:
                    continue

                overlap_ratio = (iw * ih) / A_i

                if lab_j in person_or_nude:
                    continue

                if overlap_ratio >= 0.20:
                    remove_idx.add(j)

        filtered = [
            (b, s, l)
            for k, (b, s, l) in enumerate(zip(boxes_f, scores_f, labels_f))
            if k not in remove_idx
        ]

        if filtered:
            boxes_f, scores_f, labels_f = map(list, zip(*filtered))
        else:
            boxes_f, scores_f, labels_f = [], [], []

    post_enl = [enlarge_box(box=b, scale=box_scale, img_w=w, img_h=h) for b in boxes_f]
    
    draw_annotated(image, post_enl, [s.item() for s in scores_f], labels_f, font=font).save(post_ann_out / f"{stem}.jpg", quality=jpeg_q)

    detections_post = [
        {"box": box, "label": lab, "score": float(scr.item())}
        for box, lab, scr in zip(post_enl, labels_f, scores_f)   
    ]

    with open(post_json_out / f"{stem}.json", "w") as f:
        json.dump(
            {"image": img_path.name,          
            "detections": detections_post},
            f, indent=2
        )
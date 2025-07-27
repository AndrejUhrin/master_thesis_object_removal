import json, shutil
from pathlib import Path
import sys

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parents[2]
sys.path.insert(0, str(repo_root))

import torch
from PIL import Image, ImageFont
from utils.owl import load_owl, detect_with_owl
from utils.box_mask import draw_annotated

input_dataset = repo_root / "pipeline_optimization_dataset"
base_path     = repo_root / "archived" / "run_files_archived" / "archived_results"/ "without_a_in_prompts_results"

annot15_dir = base_path / "annot_015"
json15_dir  = base_path / "json_015"

annot02_dir = base_path / "annot_02"
json02_dir  = base_path / "json_02"

annot03_dir = base_path / "annot_03"
json03_dir  = base_path / "json_03"

for d in (annot15_dir, json15_dir, annot02_dir, json02_dir, annot03_dir, json03_dir):
    d.mkdir(parents=True, exist_ok=True)

owl_threshold   = 0.15  
filter_thresh20 = 0.20    
filter_thresh30 = 0.30    
jpeg_quality    = 95

text_labels = [[
    "house number", "license plate", "person", "face",
    "religious symbol", "political symbol", "cat", "dog",
    "calendar", "paper", "framed picture", "picture",
    "poster board", "name", "sex toy", "nude image",
]]

font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=50)

all_imgs = sorted(
    p for p in input_dataset.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg"}
)

print("Loading OWLv2")
processor_owl, model_owl = load_owl()
print("Done.\n")

for idx, img_path in enumerate(all_imgs, 1):
    stem = img_path.stem
    print(f"[{idx}/{len(all_imgs)}] {stem}")

    image = Image.open(img_path).convert("RGB")

    # 1) object detection with threshold 0.15
    boxes_p, scores, labels = detect_with_owl(
        image, text_labels, processor_owl, model_owl, threshold=owl_threshold
    )
    raw_boxes = [b.tolist() for b in boxes_p]
    scores_f  = [float(s.item()) for s in scores]

    draw_annotated(image, raw_boxes, scores_f, labels, font=font) \
        .save(annot15_dir / f"{stem}.jpg", quality=jpeg_quality)

    detections15 = [
        {"box": box, "label": lab, "score": score}
        for box, lab, score in zip(raw_boxes, labels, scores_f)
    ]
    with open(json15_dir / f"{stem}.json", "w") as f:
        json.dump({"image": str(img_path), "detections": detections15}, f, indent=2)

    # 2) threshold at 0.20
    idxs_keep20 = [i for i, sc in enumerate(scores_f) if sc >= filter_thresh20]
    boxes20  = [raw_boxes[i] for i in idxs_keep20]
    labels20 = [labels[i]     for i in idxs_keep20]
    scores20 = [scores_f[i]   for i in idxs_keep20]

    draw_annotated(image, boxes20, scores20, labels20, font=font) \
        .save(annot02_dir / f"{stem}.jpg", quality=jpeg_quality)

    detections20 = [
        {"box": box, "label": lab, "score": score}
        for box, lab, score in zip(boxes20, labels20, scores20)
    ]
    with open(json02_dir / f"{stem}.json", "w") as f:
        json.dump({"image": str(img_path), "detections": detections20}, f, indent=2)

    # 3) threshold at 0.30 
    idxs_keep30 = [i for i, sc in enumerate(scores_f) if sc >= filter_thresh30]
    boxes30  = [raw_boxes[i] for i in idxs_keep30]
    labels30 = [labels[i]     for i in idxs_keep30]
    scores30 = [scores_f[i]   for i in idxs_keep30]

    draw_annotated(image, boxes30, scores30, labels30, font=font) \
        .save(annot03_dir / f"{stem}.jpg", quality=jpeg_quality)

    detections30 = [
        {"box": box, "label": lab, "score": score}
        for box, lab, score in zip(boxes30, labels30, scores30)
    ]
    with open(json03_dir / f"{stem}.json", "w") as f:
        json.dump({"image": str(img_path), "detections": detections30}, f, indent=2)

    print(
        f"{len(raw_boxes)} detections (@0.15), "
        f"{len(boxes20)} remaining (@0.20), "
        f"{len(boxes30)} remaining (@0.30)\n"
    )

print("Done running object detection only.")

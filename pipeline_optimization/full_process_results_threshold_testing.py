import json, shutil
from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent      
project_root = script_dir.parent                  
sys.path.insert(0, str(project_root))
import torch
from PIL import Image, ImageFont
from utils.owl import load_owl, detect_with_owl
from utils.box_mask import draw_annotated

input_dataset = Path("../pipeline_optimization_dataset")
base_path = Path("../results_pipeline_optimization_dataset/results_threshold_testing")

annot15_dir = base_path / "annot_015"
json15_dir = base_path / "json_015"
annot02_dir = base_path / "annot_02"
json02_dir = base_path / "json_02"

for d in (annot15_dir, json15_dir, annot02_dir, json02_dir):
    d.mkdir(parents=True, exist_ok=True)

owl_threshold = 0.15
filter_thresh = 0.20
jpeg_quality = 95

text_labels = [[
    "a house number", "a license plate", "person", "a face",
    "a religious symbol", "a political symbol", "a cat", "a dog",
    "a calendar", "a paper", "a framed picture", "a picture",
    "a poster board", "a name", "a sex toy", "a nude image",
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
    boxes_p, scores, labels = detect_with_owl(image, text_labels, processor_owl, model_owl, threshold=owl_threshold)

    raw_boxes = [b.tolist() for b in boxes_p]
    scores_f = [float(s.item()) for s in scores]

    draw_annotated(image, raw_boxes, scores_f, labels, font=font) \
        .save(annot15_dir / f"{stem}.jpg", quality=jpeg_quality)

    detections15 = [
        {"box": box, "label": lab, "score": score}
        for box, lab, score in zip(raw_boxes, labels, scores_f)
    ]
    with open(json15_dir / f"{stem}.json", "w") as f:
        json.dump({"image": str(img_path), "detections": detections15}, f, indent=2)

    # 2) filter object detection that has confidence thrshold lower than 0.2
    idxs_keep = [i for i, sc in enumerate(scores_f) if sc >= filter_thresh]
    boxes2 = [raw_boxes[i] for i in idxs_keep]
    labels2 = [labels[i] for i in idxs_keep]
    scores2 = [scores_f[i] for i in idxs_keep]

    draw_annotated(image, boxes2, scores2, labels2, font=font) \
        .save(annot02_dir / f"{stem}.jpg", quality=jpeg_quality)

    detections02 = [
        {"box": box, "label": lab, "score": score}
        for box, lab, score in zip(boxes2, labels2, scores2)
    ]
    with open(json02_dir / f"{stem}.json", "w") as f:
        json.dump({"image": str(img_path), "detections": detections02}, f, indent=2)

    print(f"{len(raw_boxes)} detections (@0.15), {len(boxes2)} remaining (@0.2)\n")

print("Done running object detection only.")

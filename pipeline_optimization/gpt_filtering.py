import json
from pathlib import Path
import sys
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
script_dir = Path(__file__).resolve().parent      
project_root = script_dir.parent                  
sys.path.insert(0, str(project_root))

from utils.clip import load_clip, classify_image
from utils.owl import load_owl, detect_with_owl
from utils.box_mask import enlarge_box, draw_annotated
from utils.gpt_function_calling import GPTInterfaceFC
from utils.template import get_template_env


input_dataset = Path("../pipeline_optimization_dataset")
base_path = Path("../post_filtering_results/gpt_postfiltering_results")

pre_raw_ann = base_path / "pre_filter_annotated"
post_ann_out = base_path / "post_filter_annotated"
gpt_ann_out = base_path / "gpt_filter_annotated"

pre_json_out = base_path / "json_results_pre_filtering"
post_json_out = base_path / "json_results_post_filtering"
gpt_json_out = base_path / "json_results_gpt_filtering"

for d in (
    pre_raw_ann, post_ann_out, gpt_ann_out, pre_json_out,
    post_json_out, gpt_json_out
):
    d.mkdir(parents=True, exist_ok=True)

n_images = None    
owl_thresh = 0.17
jpeg_q = 95
box_scale = 1.1

print("Loading models")
processor_clip, model_clip = load_clip()
processor_owl,  model_owl  = load_owl()
env = get_template_env()
gpt_fc = GPTInterfaceFC(env, model="gpt-4.1", temperature=0.7)
print("Models loaded\n")

text_labels_outside = [[
    "a house number", "a license plate", "person", "a face",
    "a religious symbol", "a political symbol", "a cat", "a dog",
]]
text_labels_inside = [[
    "a calendar", "a license plate", "a paper", "person",
    "a framed picture", "a picture", "a poster board",
    "a name", "a face", "a religious symbol", "a political symbol",
    "a sex toy", "a nude image", "a cat", "a dog"
]]
per_label_thresh = {
    "a calendar": 0.20, "a paper": 0.20, "a house number": 0.21,
    "a license plate": 0.19, "person": 0.20, "a framed picture": 0.19,
    "a picture": 0.19, "a poster board": 0.30, "a name": 0.20,
    "a face": 0.20, "a religious symbol": 0.24, "a political symbol": 0.20,
    "a sex toy": 0.23, "a nude image": 0.30, "a cat": 0.28, "a dog": 0.28
}
default_thresh = owl_thresh

font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=50)

all_imgs = sorted(
    [
        p
        for p in input_dataset.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
)
if n_images is not None:
    all_imgs = all_imgs[:n_images]
print(f"Processing {len(all_imgs)} image(s) from {input_dataset}\n")

for idx, img_path in enumerate(all_imgs, start=1):
    stem = img_path.stem
    print(f"[{idx}/{len(all_imgs)}] {img_path.name}")

    # 1) CLIP decides if the image is indoor or outdoor scene
    image = Image.open(img_path).convert("RGB")
    w, h  = image.size
    inout = classify_image(image, processor_clip, model_clip)
    labs  = text_labels_inside if inout == "an indoor scene" else text_labels_outside

    # 2) OWLv2 object detection
    boxes_p, scores, labels = detect_with_owl(image, labs, processor_owl, model_owl, threshold=owl_thresh)

    raw_boxes = [b.tolist() for b in boxes_p]

    raw_enl = [enlarge_box(box=b, scale=box_scale, img_w=w, img_h=h) for b in raw_boxes]

    draw_annotated(image, raw_enl, [s.item() for s in scores], labels, font=font).save(pre_raw_ann / f"{stem}.jpg", quality=jpeg_q)

    with open(pre_json_out / f"{stem}.json", "w") as f:
        json.dump({
            "image": img_path.name,
            "detections": [
                {"box": b, "label": l, "score": float(s.item())}
                for b, l, s in zip(raw_enl, labels, scores)
            ]
        }, f, indent=2)

    # 3) filtering detections based on threshold + logic on background prompts(a television, a window, a mirror)
    kept = [
        (b, s, l) for b, s, l in zip(raw_boxes, scores, labels)
        if s.item() >= per_label_thresh.get(l, default_thresh)
    ]
    if kept:
        boxes_f, scores_f, labels_f = map(list, zip(*kept))
    else:
        boxes_f, scores_f, labels_f = [], [], []
        print("no boxes survive score filter; skipping")

    post_enl = [enlarge_box(box=b, scale=box_scale, img_w=w, img_h=h) for b in boxes_f]
    if post_enl:
        draw_annotated(image, post_enl, [s.item() for s in scores_f], labels_f, font=font).save(post_ann_out / f"{stem}.jpg", quality=jpeg_q)

    with open(post_json_out / f"{stem}.json", "w") as f:
        json.dump({
            "image": img_path.name,
            "detections": [
                {"box": b, "label": l, "score": float(s.item())}
                for b, l, s in zip(post_enl, labels_f, scores_f)
            ]
        }, f, indent=2)

    if not post_enl:
        print("no boxes after filtering; skipping gpt filtering")
        continue

    kept_labels = [
        {"label": l, "box": b_enl, "score": s.item()}       
        for b_enl, l, s in zip(post_enl, labels_f, scores_f)
    ]

    # 4) GPT post-filtering:
    #    - Applies only to indoor scenes and labels "a picture" or "a framed picture".
    #    - For each such box, crop the image and call `gpt_fc.query_inside_fc`, which returns {"keep": True} or {"keep": False}.
    #    - If keep is True, append that detection to `final_dets`; if False, drop it.
    #    - All other labels—and any detection in an outdoor scene—skip the GPT call and are appended directly to `final_dets`.
    final_dets = []
    for det in kept_labels:
        lbl, box, score = det["label"], det["box"], det["score"]

        if inout == "an indoor scene" and lbl in {"a picture", "a framed picture"}:
            result, usage = gpt_fc.query_inside_fc(
                image=image.crop(box),
                label=lbl,
                score=score,
                box=box,
            )
            print(
                f"GPT result: {result}  | "
                f"tokens: prompt={usage['prompt_tokens']}, "
                f"completion={usage['completion_tokens']}, "
                f"total={usage['total_tokens']}"
            )
            if result["keep"]:
                final_dets.append(det)
        else:
            final_dets.append(det)
    if final_dets:
        boxes_gpt  = [d["box"] for d in final_dets]   
        labels_gpt = [d["label"] for d in final_dets]
        scores_gpt = [d["score"] for d in final_dets]
    else:
        boxes_gpt, labels_gpt, scores_gpt = [], [], []

    if boxes_gpt:                         
        draw_annotated(
            image, boxes_gpt, scores_gpt, labels_gpt, font=font
        ).save(gpt_ann_out / f"{stem}.jpg", quality=jpeg_q)

    with open(gpt_json_out / f"{stem}.json", "w") as f:
        json.dump(
            {
                "image": img_path.name,
                "detections": [
                    {"box": b, "label": l, "score": float(s)}
                    for b, l, s in zip(boxes_gpt, labels_gpt, scores_gpt)
                ],
            },
            f,
            indent=2,
        )

print("All done")


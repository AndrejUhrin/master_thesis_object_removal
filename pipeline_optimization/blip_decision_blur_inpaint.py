import json
from pathlib import Path
from typing import List
import sys
from PIL import Image, ImageFont, ImageDraw
script_dir = Path(__file__).resolve().parent      
project_root = script_dir.parent                  
sys.path.insert(0, str(project_root))

from utils.clip import load_clip, classify_image
from utils.owl import load_owl, detect_with_owl
from utils.blip import load_blip, classify_boxes
from utils.box_mask import enlarge_box,draw_annotated

input_dataset = Path("../pipeline_optimization_dataset")
base_path = Path("../inpaint_blur_results/blip_inpaint_blur_results")

prefilter_bounding_boxes_annotated = base_path / "pre_filter_annotated"

postfilter_bounding_boxes_annotated = base_path / "post_filter_annotated"

prefilter_json_boxes = base_path / "json_results_pre"
postfilter_json_boxes = base_path / "json_results_post"

blip_blur_inpaint_classification = base_path / "blip_classification_annotated"


for folder in (
    base_path,
    prefilter_bounding_boxes_annotated,
    postfilter_bounding_boxes_annotated,
    prefilter_json_boxes,
    postfilter_json_boxes,
    blip_blur_inpaint_classification
):
    folder.mkdir(parents=True, exist_ok=True)

n_images = None  
owl_thresh = 0.18
jpeg_q = 95
box_scale  = 1.10

print("Loading models")
processor_clip, model_clip = load_clip()
processor_owl, model_owl = load_owl()
processor_blip, model_blip = load_blip()
print("Models loaded")

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
    "a calendar": 0.20, "a paper": 0.20, "a house number": 0.21,
    "a license plate": 0.19, "person": 0.20, "a framed picture": 0.22,
    "a picture": 0.22, "a poster board": 0.30, "a name": 0.20,
    "a face": 0.20, "a religious symbol": 0.24, "a political symbol": 0.20,
    "a sex toy": 0.23, "a nude image": 0.30, "a cat": 0.28, "a dog": 0.28,
    "a mirror": 0.30, "a window": 0.30, "a television": 0.50
}
default_thresh = owl_thresh

font = ImageFont.truetype("/Library/fonts/Arial.ttf", size=50)

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

for idx, img_path in enumerate(all_imgs, 1):
    stem = img_path.stem
    print(f"[{idx}/{len(all_imgs)}] {img_path.name}")

    # 1) CLIP decides if the image is indoor or outdoor scene
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    inout = classify_image(image, processor_clip, model_clip)
    labs = text_labels_inside if inout == "an indoor scene" else text_labels_outside

    # 2) OWLv2 object detection
    boxes_p, scores, labels = detect_with_owl(image, labs, processor_owl, model_owl, threshold=owl_thresh)
    raw_boxes = [b.tolist() for b in boxes_p]

    draw_annotated(image, raw_boxes, [s.item() for s in scores], labels, font=font).save(prefilter_bounding_boxes_annotated / f"{stem}.jpg", quality=jpeg_q)

    detections_pre = [
        {"box": box, "label": lab, "score": float(scr.item())}
        for box, lab, scr in zip(raw_boxes, labels, scores)
    ]
    with open(prefilter_json_boxes / f"{stem}.json", "w") as f:
        json.dump({"image": img_path.name, "detections": detections_pre}, f, indent=2)

    # 3) filtering detections based on threshold + logic on background prompts(a television, a window, a mirror)
    kept = [
        (b, s, l)
        for b, s, l in zip(raw_boxes, scores, labels)
        if s.item() >= per_label_thresh.get(l, default_thresh)
    ]
    if kept:
        boxes_f, scores_f, labels_f = map(list, zip(*kept))
    else:
        boxes_f, scores_f, labels_f = [], [], []
        print("no boxes survive score filter; skipping.\n")

    if boxes_f:
        person_or_nude = {"person", "a nude image"}
        always_drop = {"a television", "a window", "a mirror"}

        remove_idx = set()                 

        for i, lab in enumerate(labels_f):
            if lab in always_drop:
                remove_idx.add(i)

        for i, (box_i, lab_i) in enumerate(zip(boxes_f, labels_f)):
            if lab_i not in ("a television", "a window"):
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
    
    draw_annotated(image, post_enl, [s.item() for s in scores_f], labels_f, font=font).save(postfilter_bounding_boxes_annotated / f"{stem}.jpg", quality=jpeg_q)

    detections_post = [
        {"box": box, "label": lab, "score": float(scr.item())}
        for box, lab, scr in zip(post_enl, labels_f, scores_f)
    ]
    with open(postfilter_json_boxes / f"{stem}.json", "w") as f:
        json.dump({"image": img_path.name, "detections": detections_post}, f, indent=2)

    if not post_enl:
        print("no boxes detected, not running BLIP.\n")
        continue
    
    # 5) BLIP-based blur/inpaint decision:
    #    - If the scene is outdoor:
    #       - license plate = blur
    #       - everything other label = inpaint
    #    - If the scene is indoor:
    #       - always inpaint any “person”
    #       - for all other boxes, crop & batch‑classify with BLIP (`classify_boxes`) which returns “inpaint” or “blur for each region

    actions: List[str] = []

    if inout != "an indoor scene":
        for lbl in labels_f:
            if lbl == "a license plate" or lbl == "license plate":
                actions.append("blur")
            else:
                actions.append("inpaint")
    else:
        blip_boxes = []
        blip_indices = []
        for i, lbl in enumerate(labels_f):
            if lbl == "person":
                actions.append("inpaint")
            else:
                blip_indices.append(i)
                blip_boxes.append(post_enl[i])
                actions.append(None)  

        if blip_boxes:
            blip_results = classify_boxes(
                image,
                blip_boxes,
                processor_blip,
                model_blip,
                expand_factor=1.4,
                min_crop_size=250
            )
            for idx_rel, act in enumerate(blip_results):
                idx_abs = blip_indices[idx_rel]
                actions[idx_abs] = act

    for i in range(len(actions)):
        if actions[i] is None:
            actions[i] = "blur"

    class_annot = image.copy() 
    draw_cls = ImageDraw.Draw(class_annot)
    for (x0, y0, x1, y1), lbl, act in zip(post_enl, labels_f, actions):
        color = "blue" if act == "inpaint" else "red"
        draw_cls.rectangle([x0, y0, x1, y1], outline=color, width=4)
        tag = f"{lbl.replace('_', ' ')}, {act}"
        tw, th = draw_cls.textbbox((0, 0), tag, font=font)[2:]
        draw_cls.rectangle([x0, y0 - th - 4, x0 + tw + 4, y0], fill="white")
        draw_cls.text((x0 + 2, y0 - th - 2), tag, fill=color, font=font)

    class_annot.convert("RGB").save(blip_blur_inpaint_classification / f"{stem}.jpg", quality=jpeg_q)
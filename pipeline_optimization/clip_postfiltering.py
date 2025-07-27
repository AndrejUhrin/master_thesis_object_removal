import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import sys
script_dir = Path(__file__).resolve().parent      
project_root = script_dir.parent                  
sys.path.insert(0, str(project_root))

from utils.clip import load_clip, classify_image
from utils.owl import load_owl, detect_with_owl
from utils.box_mask import enlarge_box, draw_annotated

input_dataset = Path("../pipeline_optimization_dataset")
base_path = Path("../post_filtering_and_blur_inpaint_results/clip_postfiltering_results")

pre_raw_ann = base_path / "pre_filter_annotated"
post_ann_out = base_path / "post_filter_annotated"
clip_ann_out = base_path / "post_clip_annotated"

pre_json_out = base_path / "json_results_pre_filtering"
post_json_out = base_path / "json_results_post_filtering"
clip_json_out = base_path / "json_results_clip_filtering"

for d in (
    pre_raw_ann,
    post_ann_out,
    clip_ann_out,
    pre_json_out,
    post_json_out,
    clip_json_out,
):
    d.mkdir(parents=True, exist_ok=True)

n_images = None
owl_thresh = 0.18
jpeg_q = 95
box_scale = 1.15

font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=50)
clip_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=40)

print("Loading models")
processor_clip, model_clip = load_clip()
processor_owl,  model_owl  = load_owl()
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
]]
per_label_thresh = {
    "a calendar": 0.20, "a paper": 0.20, "a house number": 0.21,
    "a license plate": 0.19, "person": 0.20, "a framed picture": 0.22,
    "a picture": 0.22, "a poster board": 0.30, "a name": 0.20,
    "a face": 0.20, "a religious symbol": 0.24, "a political symbol": 0.20,
    "a sex toy": 0.23, "a nude image": 0.30, "a cat": 0.28, "a dog": 0.28,
}
def_thresh = owl_thresh

clip_labels = {
    "a house number", "a political symbol", "a family picture", "a picture frame",
    "a canvas", "a picture", "a paper",
}

# running CLIP zero‑shot classification on the cropped region, returns (best_prompt_index, probabilities)
def clip_multi(crop, prompts):
    inputs = processor_clip(text=prompts, images=crop, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model_clip(**inputs).logits_per_image.squeeze(0)
        probs  = torch.softmax(logits, dim=0)
    probs_list = probs.tolist()
    return int(torch.argmax(probs).item()), probs_list

# defining positive and negative labels for CLIP postfiltering on the cropped region
def get_clip_prompts(label: str):
    if label == "a house number":
        return ["a digits on a wall", "a house number", "a number"], ["a blank wall without digits", "a door or window", "decoration","a street sign with text"]
    elif label == "a political symbol":
        return ["a flag", "a political symbol"], ["a traffic sign", "an advertisement poster"]
    elif label == "a framed picture":
        return ["a framed picture", "a photograph"], ["a pianting", "an artwork", "a wooden board", "a blank wall", "a door", "a window", "a television"]
    elif label == "a picture":
        return ["a picture", "a photograph", "a framed picture"], ["a pianting", "an artwork","a wooden board", "a blank wall", "a door", "a window", "a television"]
    elif label == "a paper":
        return ["a paper"], ["a book", "a folder"]
    return [], []

all_imgs = sorted(
    [
        p
        for p in input_dataset.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg"}
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
    w, h  = image.size
    inout = classify_image(image, processor_clip, model_clip)
    labs  = text_labels_inside if inout == "an indoor scene" else text_labels_outside

    # 2) OWLv2 object detection
    boxes_p, scores, labels = detect_with_owl(image, labs, processor_owl, model_owl, threshold=owl_thresh)
    raw_boxes = [b.tolist() for b in boxes_p]
    enlarged_boxes = [enlarge_box(b, box_scale, img_w=w, img_h=h) for b in raw_boxes]

    draw_annotated(image, enlarged_boxes, [s.item() for s in scores], labels, font=font).save(pre_raw_ann / f"{stem}.jpg", quality=jpeg_q)

    with open(pre_json_out / f"{stem}.json", "w") as f:
        json.dump(
            {
                "image": img_path.name,
                "detections": [
                    {"box": b, "label": l, "score": float(s.item())}
                    for b, l, s in zip(enlarged_boxes, labels, scores)
                ],
            },
            f,
            indent=2,
        )

    # 3) filtering detections based on threshold + logic on background prompts(a television, a window, a mirror)
    kept = [
        (b, s.item(), l)
        for b, s, l in zip(enlarged_boxes, scores, labels)
        if s.item() >= per_label_thresh.get(l, def_thresh)
    ]
    if not kept:
        print("No boxes kept after threshold\n")
        continue

    kept_dicts = [{"box": box, "label": label, "score": score} for box, score, label in kept]
    
    draw_annotated(image, [d["box"] for d in kept_dicts], [d["score"] for d in kept_dicts], [d["label"] for d in kept_dicts], font=font,).save(post_ann_out / f"{stem}.jpg", quality=jpeg_q)

    with open(post_json_out / f"{stem}.json", "w") as f:
        json.dump({"image": img_path.name, "detections": kept_dicts}, f, indent=2)

    # CLIP postfiltering step
    kept_clip = []          
    for det in kept_dicts:
        lbl = det["label"]
        det["clip_pass"] = True            

        if lbl in clip_labels:
            if lbl in {"a house number", "a political symbol"} and inout == "an outdoor scene":
                x0, y0, x1, y1 = map(int, det["box"])
                crop = image.crop((x0, y0, x1, y1))
                pos, neg = get_clip_prompts(lbl)
                prompts = pos + neg
                if prompts:
                    win_idx, confs = clip_multi(crop, prompts)
                    det["clip_pass"] = win_idx < len(pos)
            elif lbl not in {"a house number", "a political symbol"}:
                x0, y0, x1, y1 = map(int, det["box"])
                crop = image.crop((x0, y0, x1, y1))
                pos, neg = get_clip_prompts(lbl)
                prompts = pos + neg
                if prompts:
                    win_idx, confs = clip_multi(crop, prompts)
                    det["clip_pass"] = win_idx < len(pos)

        if det["clip_pass"]:
            kept_clip.append(det)          

    with open(clip_json_out / f"{stem}.json", "w") as f:
        json.dump({"image": img_path.name, "detections": kept_clip}, f, indent=2)

    if kept_clip:  
        draw_annotated(image, [d["box"]  for d in kept_clip], [d["score"] for d in kept_clip], [d["label"] for d in kept_clip], font=clip_font,).save(clip_ann_out / f"{stem}.jpg", quality=jpeg_q)

print("\nProcessing complete.")

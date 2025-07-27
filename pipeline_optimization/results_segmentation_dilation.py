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
from utils.sam import load_sam, segment_with_sam, post_process_sam_masks
from utils.mask import overlay_masks_on_image, create_black_white_mask
from utils.box_mask import enlarge_box, draw_annotated


input_dir = Path("../pipeline_optimization_dataset")
base_out = Path("../results_pipeline_optimization_dataset/results_segmentation_dilation")

pre_raw_ann = base_out / "pre_filter_annotated"

post_ann_out = base_out / "post_filter_annotated"

pre_json_out = base_out / "json_results_pre"
post_json_out = base_out / "json_results_post"

seg_dil_20 = base_out / "segmentation_overlay_dilation20"
seg_mask_dil_20 = base_out / "segmentation_masks_bw_dilation20"
seg_dil_30 = base_out / "segmentation_overlay_dilation30"
seg_mask_dil_30 = base_out / "segmentation_masks_bw_dilation30"
seg_dil_40 = base_out / "segmentation_overlay_dilation40"
seg_mask_dil_40 = base_out / "segmentation_masks_bw_dilation40"
seg_dil_50  = base_out / "segmentation_overlay_dilation50"
seg_mask_dil_50 = base_out / "segmentation_masks_bw_dilation50"
seg_dil_55  = base_out / "segmentation_overlay_dilation55"
seg_mask_dil_55 = base_out / "segmentation_masks_bw_dilation55"
seg_dil_60  = base_out / "segmentation_overlay_dilation60"
seg_mask_dil_60 = base_out / "segmentation_masks_bw_dilation60"


for d in (
    pre_raw_ann, post_ann_out,
    pre_json_out, post_json_out,    
    seg_dil_20, seg_mask_dil_20,
    seg_dil_30, seg_mask_dil_30,
    seg_dil_40, seg_mask_dil_40,
    seg_dil_50, seg_mask_dil_50,
    seg_dil_55, seg_mask_dil_55,
    seg_dil_60, seg_mask_dil_60,
):
    d.mkdir(parents=True, exist_ok=True)

n_images = 200      
owl_thresh = 0.18
jpeg_q = 95
box_scale = 1.1

print("Loading models")
processor_clip, model_clip = load_clip()      
processor_owl, model_owl  = load_owl()
processor_sam, model_sam  = load_sam()
print("Models loaded")

text_labels_outside = [[
    "a house number", "a license plate", "person", "a face",
    "a religious symbol", "a political symbol", "a cat", "a dog",
]]

text_labels_inside = [[
    "a calendar", "a license plate", "a paper", "person",
    "a framed picture", "a canvas", "a picture", "a poster board",
    "a name", "a face", "a religious symbol", "a political symbol",
    "a sex toy", "a nude image", "a cat", "a dog",
]]

per_label_thresh = {
    "a calendar": 0.20, "a paper": 0.20, "a house number": 0.21,
    "a license plate": 0.19, "person": 0.20, "a framed picture": 0.22,
    "a picture": 0.22, "a poster board": 0.30, "a name": 0.20,
    "a face": 0.20, "a religious symbol": 0.24, "a political symbol": 0.20,
    "a sex toy": 0.23, "a nude image": 0.30, "a cat": 0.24,
    "a dog": 0.24, "a canvas": 0.22,
}
default_thresh = owl_thresh

font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=50)

all_imgs = sorted(
    p for p in input_dir.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)
if n_images is not None:
    all_imgs = all_imgs[:n_images]
print(f"Processing {len(all_imgs)} image(s) from {input_dir}\n")

for idx, img_path in enumerate(all_imgs, 1):
    stem = img_path.stem
    print(f"[{idx}/{len(all_imgs)}] {img_path.name}")

    # 1) CLIP decides if the image is indoor or outdoor scene
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    inout = classify_image(image, processor_clip, model_clip)
    labels = text_labels_inside if inout == "an indoor scene" else text_labels_outside

    # 2) OWLv2 object detection
    boxes_p, scores, labels_pred = detect_with_owl(image, labels, processor_owl, model_owl, threshold=owl_thresh)

    raw_boxes = [b.tolist() for b in boxes_p]

    raw_enl = [enlarge_box(b, box_scale, img_w=w, img_h=h) for b in raw_boxes]

    draw_annotated(image, raw_enl, [s.item() for s in scores], labels_pred, font=font).save(pre_raw_ann / f"{stem}.jpg", quality=jpeg_q)

    detections_pre = [
        {"box": box, "label": lab, "score": float(scr.item())}
        for box, lab, scr in zip(raw_enl, labels_pred, scores)
    ]
    (pre_json_out / f"{stem}.json").write_text(json.dumps(
        {"image": img_path.name, "detections": detections_pre}, indent=2
    ))

    # 3) filtering detections based on threshold + logic on background prompts(a television, a window, a mirror)
    kept = [
        (b, s, l)
        for b, s, l in zip(raw_boxes, scores, labels_pred)
        if s.item() >= per_label_thresh.get(l, default_thresh)
    ]
    boxes_f, scores_f, labels_f = ([], [], []) if not kept else map(list, zip(*kept))

    post_enl = [enlarge_box(b, box_scale, img_w=w, img_h=h) for b in boxes_f]

    draw_annotated(image, post_enl, [s.item() for s in scores_f], labels_f, font=font).save(post_ann_out / f"{stem}.jpg", quality=jpeg_q)

    detections_post = [
        {"box": box, "label": lab, "score": float(scr.item())}
        for box, lab, scr in zip(post_enl, labels_f, scores_f)
    ]
    (post_json_out / f"{stem}.json").write_text(json.dumps(
        {"image": img_path.name, "detections": detections_post}, indent=2
    ))

    # 4) Segmentation with different dilation pixel sizes
    if post_enl:
        tb = torch.tensor(post_enl, dtype=torch.float32).unsqueeze(0)
        outs, ins = segment_with_sam(image, tb, processor_sam, model_sam)
        masks = post_process_sam_masks(outs, processor_sam, ins)[0]

        for dil, seg_dir, mask_dir in [
            (20, seg_dil_20, seg_mask_dil_20),
            (30, seg_dil_30, seg_mask_dil_30),
            (40, seg_dil_40, seg_mask_dil_40),
            (50, seg_dil_50, seg_mask_dil_50),
            (55, seg_dil_55, seg_mask_dil_55),
            (60, seg_dil_60, seg_mask_dil_60),
        ]:
            overlay_masks_on_image(image, masks, overlay_color=(255, 0, 0, 128), dilation_px=dil).save(seg_dir / f"{stem}.jpg", quality=jpeg_q)

            create_black_white_mask(masks, threshold=0.5, combine=True, dilation_px=dil).save(mask_dir / f"{stem}.png", optimize=True)
    else:
        print(f"no post-filter boxes; skipping segmentation for {stem}.\n")

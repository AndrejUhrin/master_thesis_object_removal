import json, shutil
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
script_dir = Path(__file__).resolve().parent      
project_root = script_dir.parent                  
sys.path.insert(0, str(project_root))
from utils.clip import load_clip, classify_image
from utils.owl import load_owl, detect_with_owl
from utils.sam import load_sam, segment_with_sam, post_process_sam_masks
from utils.mask import create_black_white_mask
from utils.box_mask import enlarge_box, draw_annotated
from utils.opencv_canny import analyze_segmentation_edges
from lama.runner import run_lama
from utils.bluring import adaptive_blur
from utils.resize   import resize_long_side 

input_dir = Path("../final_evaluation_dataset")
base_path = Path("../results_on_unseen_dataset")

lama_model_dir = Path("../lama/big-lama")

tmp_dir = base_path / "tmp_staging"
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir(parents=True, exist_ok=True)

out_dir = base_path / "final_inpainted_image"

pre_raw_ann = base_path / "pre_filter_annotated"
post_ann_out = base_path / "post_filter_annotated"
images_no_detection = base_path / "no_object_removal"
pre_json_out = base_path / "json_results_pre"
post_json_out = base_path / "json_results_post"

for d in (pre_raw_ann, post_ann_out,
          images_no_detection,
          pre_json_out, post_json_out,
          out_dir):
    d.mkdir(parents=True, exist_ok=True)

seg_mask_bw = {
    d: base_path / f"segmentation_masks_bw_dilation{d}"
    for d in (0, 10, 30, 55)
}

blur_dir  = base_path / "masks_blur"
inpaint_dir = base_path / "masks_inpaint"

for d in seg_mask_bw.values():
    d.mkdir(parents=True, exist_ok=True)

for d in (0, 10, 30, 55):
    (blur_dir  / f"dilation{d}").mkdir(parents=True, exist_ok=True)
    (inpaint_dir / f"dilation{d}").mkdir(parents=True, exist_ok=True)

n_images = None    
owl_thresh  = 0.18
jpeg_q = 95
target_long = 2048
box_scale = 1.1

print("Loading models…")
processor_clip, model_clip = load_clip()
processor_owl,  model_owl  = load_owl()
processor_sam,  model_sam  = load_sam()
print("Models loaded\n")

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
    boxes_p, scores, labels = detect_with_owl(image, labs, processor_owl, model_owl, threshold=owl_thresh)
    raw_boxes = [b.tolist() for b in boxes_p]

    draw_annotated(image, raw_boxes, [float(s.item()) for s in scores], labels, font=font).save(pre_raw_ann / f"{stem}.jpg", quality=jpeg_q)

    detections = [
        {"box": box, "label": lab, "score": float(scr.item())}
        for box, lab, scr in zip(raw_boxes, labels, scores)
    ]
    with open(pre_json_out / f"{stem}.json", "w") as f:
        json.dump(
            {"image": str(img_path.relative_to(input_dir)), "detections": detections},
            f, indent=2
        )
    # 3) filtering detections based on threshold + logic on background prompts(a television, a window, a mirror)
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
        person_or_nude = {"person", "a nude image", "nude image"}
        always_drop    = {"a television", "a window", "a mirror"}

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

    if not post_enl:
        resized_img = resize_long_side(image, target_long)
        resized_img.save(images_no_detection / f"{stem}.jpg", quality=jpeg_q)
        print("no boxes detected; saved resized image only.\n")
        continue

    # 4) running SAM segmentation on postfiltered bounding boxes
    print("Running SAM segmentation")
    tb = torch.tensor(post_enl, dtype=torch.float32).unsqueeze(0)
    outs, ins = segment_with_sam(image, tb, processor_sam, model_sam)
    masks_from_sam = post_process_sam_masks(outs, processor_sam, ins)[0]

    is_outdoor = (inout != "an indoor scene")
    always_inpaint_indoor = {"person", "a religious symbol", "a political symbol"}

    sam_bool_masks = [(np.array(m) > 0.5).astype(np.uint8) for m in masks_from_sam]

    print(f"Generating and saving {len(masks_from_sam)} masks "
          f"for {len(seg_mask_bw)} dilation levels")
    for dpx in seg_mask_bw.keys():
        bw_combined_mask = create_black_white_mask(
            masks_from_sam,
            threshold=0.5,
            combine=True,
            dilation_px=dpx
        )
        bw_combined_mask.save(seg_mask_bw[dpx] / f"{stem}.png", optimize=True)

    # 5) Canny-based  classification for inpainting and blurring on segment
    print("Loading 10px mask for Canny analysis")
    mask_path_10px = seg_mask_bw[10] / f"{stem}.png"
    bw10_cv = cv2.imread(str(mask_path_10px), cv2.IMREAD_GRAYSCALE)

    if bw10_cv is None:
        print(f"ERROR: Could not read 10px mask at {mask_path_10px}. Skipping Canny.")
        continue

    mask10_binary = (bw10_cv > 127).astype(np.uint8)
    contours, _ = cv2.findContours(mask10_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} separate objects in mask to classify.")

    classified_contours = []
    image_for_canny = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for i, cnt in enumerate(contours):
        single_mask = np.zeros_like(mask10_binary)
        cv2.drawContours(single_mask, [cnt], -1, color=1, thickness=-1)

        overlaps = [
            np.logical_and(single_mask, sam_bool_masks[j]).sum()
            for j in range(len(sam_bool_masks))
        ]
        obj_idx = int(np.argmax(overlaps))
        lbl = labels_f[obj_idx]

        if is_outdoor:
            action = "blur" if lbl == "a license plate" else "inpaint"
            edge_ratio = float("nan")
        else:
            if lbl in always_inpaint_indoor:
                action, edge_ratio = "inpaint", float("nan")
            else:
                result = analyze_segmentation_edges(
                    image_for_canny,
                    single_mask.astype(bool),
                    margin=60,
                    edge_threshold=0.015
                )[0]
                action = result["action"]        
                edge_ratio = result["edge_ratio"]    

        classified_contours.append({
            "contour" : cnt,
            "action" : action,
            "edge_ratio" : edge_ratio
        })

        er_str = f"{edge_ratio:.3f}" if edge_ratio == edge_ratio else "—"


    # 6) Creates 2 sets of masks one for blurring and one for inpainting masks based on canny classification
    print("Re-creating final masks for all dilation levels")
    dilations = seg_mask_bw.keys()
    final_blur_masks = {d: np.zeros_like(mask10_binary, dtype=np.uint8) for d in dilations}

    final_inpnt_masks = {d: np.zeros_like(mask10_binary, dtype=np.uint8) for d in dilations}

    erosion_kernel = np.ones((2 * 10 + 1, 2 * 10 + 1), np.uint8)

    for item in classified_contours:
        cnt = item['contour']
        action = item['action']

        component_10px = np.zeros_like(mask10_binary)
        cv2.drawContours(component_10px, [cnt], -1, color=1, thickness=-1)
        component_0px  = cv2.erode(component_10px, erosion_kernel, iterations=1)

        for dpx in dilations:
            if dpx == 0:
                final_shape = component_0px
            else:
                kernel = np.ones((2 * dpx + 1, 2 * dpx + 1), np.uint8)
                final_shape = cv2.dilate(component_0px, kernel, iterations=1)

            if action == "blur":
                final_blur_masks[dpx] = np.maximum(final_blur_masks[dpx], final_shape)
            else:  
                final_inpnt_masks[dpx] = np.maximum(final_inpnt_masks[dpx], final_shape)

    saved_any = False                         
    for dpx in dilations:
        blur_mask = final_blur_masks[dpx]
        paint_mask = final_inpnt_masks[dpx]

        if blur_mask.any():                    
            cv2.imwrite(
                str(blur_dir / f"dilation{dpx}" / f"{stem}.png"),
                blur_mask * 255
            )
            saved_any = True

        if paint_mask.any():
            cv2.imwrite(
                str(inpaint_dir / f"dilation{dpx}" / f"{stem}.png"),
                paint_mask * 255
            )
            saved_any = True

    if saved_any:
        print("Saved final blur/inpaint masks.")
    else:
        print("No blur or inpaint masks contained data; nothing saved.")

    # 7) Takes the crated blur and inpaint masks, resizes them with also the original input image, and then performs inpainting/blurring to create the final image
    paint_dpx = 55 if inout == "an indoor scene" else 30   
    blur_dpx  = 0                                          

    paint_mask_path = inpaint_dir / f"dilation{paint_dpx}" / f"{stem}.png"
    blur_mask_path  = blur_dir   / f"dilation{blur_dpx}"  / f"{stem}.png"
    tmp_img  = tmp_dir / f"{stem}.png"
    tmp_mask = tmp_dir / f"{stem}_mask.png"
    tmp_blur = tmp_dir / f"{stem}_blur.png"

    img_rs = resize_long_side(image, target_long)          
    img_rs.save(tmp_img, "PNG")

    if paint_mask_path.exists():
        paint_mask = cv2.imread(str(paint_mask_path), cv2.IMREAD_GRAYSCALE)
        mask_rs = resize_long_side(paint_mask, target_long, nearest=True)
        cv2.imwrite(str(tmp_mask), mask_rs)

        try:
            result_png = run_lama(
                image_path=tmp_img,
                mask_path=tmp_mask,
                model_dir=lama_model_dir,
                out_dir=tmp_dir,
            )
            result_bgr = cv2.imread(str(result_png), cv2.IMREAD_COLOR)
            if result_bgr is None:
                print("Could not read LaMa output; using original.\n")

                final_bgr = cv2.cvtColor(np.array(img_rs), cv2.COLOR_RGB2BGR)
            else:
                final_bgr = result_bgr
        except Exception as e:
            print(f"LaMa failed ({e}); using original.\n")
            final_bgr = cv2.cvtColor(np.array(img_rs), cv2.COLOR_RGB2BGR)
    else:
        print("No in‐paint mask; skipping in‐paint.\n")
        final_bgr = cv2.cvtColor(np.array(img_rs), cv2.COLOR_RGB2BGR)


    if blur_mask_path.exists():
        blur_mask = cv2.imread(str(blur_mask_path), cv2.IMREAD_GRAYSCALE)
        blur_rs = resize_long_side(blur_mask, target_long, nearest=True)
        cv2.imwrite(str(tmp_blur), blur_rs)  
        final_bgr = adaptive_blur(final_bgr, blur_rs, strength=0.6)
    else:
        print("No blur mask; skipping blur.\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.jpg"
    cv2.imwrite(str(out_path), final_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
    print(f"Final composite saved{out_path}\n")

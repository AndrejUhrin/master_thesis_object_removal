
import json, shutil
from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent      
project_root = script_dir.parent                  
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw

from utils.owl   import load_owl, detect_with_owl
from utils.sam   import load_sam, segment_with_sam, post_process_sam_masks
from utils.mask  import overlay_masks_on_image, create_black_white_mask
from utils.box_mask import draw_annotated
from utils.resize import resize_long_side
from lama.runner import run_lama

input_dataset = Path("../pipeline_optimization_dataset")
base_path = Path("../results_pipeline_optimization_dataset/results_initial_pipeline")

object_detection_annotated = base_path / "bounding_box_annotated"
object_detection_json_results = base_path / "json_results"
images_no_detection = base_path / "no_object_removal"
segmentation_overlay = base_path / "segmentation_overlay_no_dilation"
segmentation_mask_dir = base_path / "segmentation_masks_bw_no_dilation"
final_inpainted_dir = base_path / "final_inpainted_image"

for d in (
    object_detection_annotated,
    object_detection_json_results,
    images_no_detection,
    segmentation_overlay,
    segmentation_mask_dir,
    final_inpainted_dir,
):
    d.mkdir(parents=True, exist_ok=True)


tmp_dir = base_path / "tmp_staging"
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir()


lama_model_dir = Path("../lama/big-lama")   

n_images = None     
owl_thresh = 0.3
jpeg_q = 95
target_long = 2048

print("Loading models")
processor_owl,  model_owl  = load_owl()
processor_sam,  model_sam  = load_sam()
print("Models loaded\n")

# labels used for OWLv2 for object detection
text_labels = [[
    "a house number", 
    "a license plate", 
    "person", 
    "a face",
    "a religious symbol", 
    "a political symbol", 
    "a cat", 
    "a dog",
    "a calendar", 
    "a paper", 
    "a framed picture", 
    "a picture",
    "a poster board", 
    "a name", 
    "a sex toy", 
    "a nude image",
]]

# font size for annotated images
font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=50)

all_imgs = sorted(
    p for p in input_dataset.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg"}
)
if n_images:
    all_imgs = all_imgs[:n_images]
print(f"Processing {len(all_imgs)} image(s) from {input_dataset}\n")

# start of the full main process
for idx, img_path in enumerate(all_imgs, 1):
    stem = img_path.stem
    print(f"[{idx}/{len(all_imgs)}] {img_path.relative_to(input_dataset)}")

    for p in tmp_dir.iterdir():
        (shutil.rmtree(p) if p.is_dir() else p.unlink())

    # 1) loading of the images
    image = Image.open(img_path).convert("RGB")
    w, h  = image.size

    # 2) Object detection with OWLv2
    boxes_p, scores, labels = detect_with_owl(image, text_labels, processor_owl, model_owl, threshold=owl_thresh)

    raw_boxes = [b.tolist() for b in boxes_p]

    # 3) saving annotated results for visual check and json files for analysis of the results
    draw_annotated(image, raw_boxes, [float(s.item()) for s in scores], labels, font=font).save(object_detection_annotated / f"{stem}.jpg", quality=jpeg_q)

    detections = [
        {"box": box, "label": lab, "score": float(scr.item())}
        for box, lab, scr in zip(raw_boxes, labels, scores)
    ]
    with open(object_detection_json_results / f"{stem}.json", "w") as f:
        json.dump(
            {"image": str(img_path.relative_to(input_dataset)), "detections": detections},
            f, indent=2
        )

    # Case 1: if nothing gets detected just store the final results but resized so they are consistent with the inpainted images
    if not raw_boxes:
        img_np = np.array(image)                     
        img_rs = resize_long_side(img_np, target_long)  
        Image.fromarray(img_rs).save(images_no_detection / f"{stem}.jpg",quality=jpeg_q)        
        print("no boxes detected; saved resized image only.\n")
        continue

    # Case 2: objects detected proceed with the process of SAM and LaMa
    tb = torch.tensor(raw_boxes, dtype=torch.float32).unsqueeze(0)
    outs, ins = segment_with_sam(image, tb, processor_sam, model_sam)
    masks = post_process_sam_masks(outs, processor_sam, ins)[0]

    # save visual overlay for visual check and black and white masks for inpainting
    overlay_masks_on_image(image, masks, overlay_color=(255, 0, 0, 128), dilation_px=0).save(segmentation_overlay / f"{stem}.jpg", quality=jpeg_q)

    bw_mask_path = segmentation_mask_dir / f"{stem}.png"
    create_black_white_mask(masks, threshold=0.5, combine=True, dilation_px=0).save(bw_mask_path, optimize=True)

    # resize image & mask to target_long for LaMa, since original image size would take too long to process
    img_np = np.array(image)                    
    img_rs = resize_long_side(img_np, target_long)
    mask_rs = resize_long_side(cv2.imread(str(bw_mask_path), cv2.IMREAD_GRAYSCALE), target_long, nearest=True)

    # both need to get stored for to a temp folder, since LaMa can do the inpainting only if the images are stored on a disk
    tmp_img  = tmp_dir / f"{stem}.png"
    tmp_mask = tmp_dir / f"{stem}_mask.png"
    Image.fromarray(img_rs).save(tmp_img, "PNG")
    cv2.imwrite(str(tmp_mask), mask_rs)

    # inpainting using LaMa on resized images and masks
    try:
        result_png = run_lama(
            image_path=tmp_img,
            mask_path=tmp_mask,
            model_dir=lama_model_dir,
            out_dir=tmp_dir,
        )
    except Exception as e:
        print(f"LaMa failed: {e}\n")
        continue

    # store final jpg
    result_img = Image.open(result_png).convert("RGB")
    result_img.save(final_inpainted_dir / f"{stem}.jpg", quality=jpeg_q)
    print("in-painted image saved.\n")

print("All done!")

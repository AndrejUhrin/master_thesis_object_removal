import shutil
from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent      
project_root = script_dir.parent                  
sys.path.insert(0, str(project_root))
import cv2
import numpy as np
from PIL import Image
from utils.resize import resize_long_side
from lama.runner  import run_lama

input_dataset = Path("../pipeline_optimization_dataset")
base_path = Path("../results_pipeline_optimization_dataset/results_3_segmentation_dilation")

segmentation_mask_dir = base_path / "segmentation_masks_bw_dilation60"  
final_inpainted_dir = base_path / "inpainted_mask_dilation_60"              

final_inpainted_dir.mkdir(parents=True, exist_ok=True)

tmp_dir = base_path / "tmp_staging"
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir()

lama_model_dir = Path("../lama/big-lama")

jpeg_q = 95
target_long = 2048

all_imgs = sorted(
    p for p in input_dataset.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg"}
)
print(f"Processing {len(all_imgs)} image(s) from {input_dataset}\n")


for idx, img_path in enumerate(all_imgs, 1):
    stem = img_path.stem
    out_path = final_inpainted_dir / f"{stem}.jpg"

    if out_path.exists():
        print(f"[{idx}/{len(all_imgs)}] {stem}.jpg already exists — skipping.\n")
        continue

    print(f"[{idx}/{len(all_imgs)}] {img_path.relative_to(input_dataset)}")

    for p in tmp_dir.iterdir():
        (shutil.rmtree(p) if p.is_dir() else p.unlink())

    image = Image.open(img_path).convert("RGB")
    mask_path = segmentation_mask_dir / f"{stem}.png"
    if not mask_path.exists():
        print("mask not found, skipping.\n")
        continue

    img_np = np.array(image)
    mask_np = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    img_rs = resize_long_side(img_np,  target_long)
    mask_rs = resize_long_side(mask_np, target_long, nearest=True)

    tmp_img  = tmp_dir / f"{stem}.png"
    tmp_mask = tmp_dir / f"{stem}_mask.png"
    Image.fromarray(img_rs).save(tmp_img, "PNG")
    cv2.imwrite(str(tmp_mask), mask_rs)

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

    Image.open(result_png).convert("RGB").save(
        final_inpainted_dir / f"{stem}.jpg",
        quality=jpeg_q
    )
    print("in-painted image saved.\n")

print("All done!")

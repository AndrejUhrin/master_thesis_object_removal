import csv
from pathlib import Path
from PIL import Image
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from utils.clip import load_clip, classify_image

input_dir   = Path("../pipeline_optimization_dataset")           
output_csv  = Path("../results_pipeline_optimization_dataset/clip_indoor_outdoor_results") / "clip_scene_results.csv"
n_images    = None                                               

print("Loading CLIP")
processor_clip, model_clip = load_clip()
print("CLIP loaded.\n")

all_imgs = sorted([
    p for p in input_dir.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])
if n_images:
    all_imgs = all_imgs[:n_images]
print(f"Classifying {len(all_imgs)} images …\n")

output_csv.parent.mkdir(parents=True, exist_ok=True)

with output_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "scene"])           

    for idx, img_path in enumerate(all_imgs, 1):
        image = Image.open(img_path).convert("RGB")
        inout = classify_image(image, processor_clip, model_clip)
        scene = "indoor" if inout == "an indoor scene" else "outdoor"

        writer.writerow([img_path.name, scene])
        print(f"[{idx}/{len(all_imgs)}] {img_path.name}: {scene}")

print(f"\nDone. Results saved to: {output_csv}")

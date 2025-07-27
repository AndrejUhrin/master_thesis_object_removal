import torch
from PIL import Image
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
from typing import List, Tuple

question_promopt = ("Is the object hanging on a wall? Yes or no.")

def load_blip():
    model_name: str = "Salesforce/instructblip-flan-t5-xl"
    processor = InstructBlipProcessor.from_pretrained(model_name)
    model = InstructBlipForConditionalGeneration.from_pretrained(model_name).eval()
    torch.set_grad_enabled(False)

    _ = model.generate(
        **processor(images=[Image.new("RGB", (224, 224))],text=["warm-up"],return_tensors="pt"),
        max_new_tokens=1
    )

    return processor, model

def classify_boxes(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    processor: InstructBlipProcessor,
    model: InstructBlipForConditionalGeneration,
    *,
    expand_factor: float = 1.4,
    min_crop_size: int = 250,
) -> List[str]:

    def expand_bbox(
        box: Tuple[int, int, int, int],
        *,
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        pad_x = (expand_factor - 1.0) * w / 2.0
        pad_y = (expand_factor - 1.0) * h / 2.0
        x0e, y0e, x1e, y1e = x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y
        new_w, new_h = x1e - x0e, y1e - y0e

        if new_w < min_crop_size:
            extra = (min_crop_size - new_w) / 2.0
            x0e, x1e = x0e - extra, x1e + extra
        if new_h < min_crop_size:
            extra = (min_crop_size - new_h) / 2.0
            y0e, y1e = y0e - extra, y1e + extra

        x0e, y0e = int(max(0, x0e)), int(max(0, y0e))
        x1e, y1e = int(min(image_width, x1e)), int(min(image_height, y1e))
        return x0e, y0e, x1e, y1e

    img_w, img_h = image.size
    crops: List[Image.Image] = []

    for box in boxes:
        x0e, y0e, x1e, y1e = expand_bbox(box, image_width=img_w, image_height=img_h)
        crop = image.crop((x0e, y0e, x1e, y1e)).resize((224, 224), Image.BILINEAR)
        crops.append(crop)

    if not crops:
        return []

    prompts = [question_promopt] * len(crops)
    inputs = processor(images=crops, text=prompts, return_tensors="pt", padding=True)
    out_ids = model.generate(**inputs, max_new_tokens=1)
    answers = processor.batch_decode(out_ids, skip_special_tokens=True)

    return ["inpaint" if a.strip().lower().startswith("yes") else "blur" for a in answers]

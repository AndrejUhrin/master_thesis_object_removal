from pathlib import Path
from typing import Union, Tuple, List
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def load_clip() -> Tuple[CLIPProcessor, CLIPModel]:

    model_name: str = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)  
    return processor, model


def classify_image(
    image_input: Union[str, Path, Image.Image],
    processor: CLIPProcessor,
    model: CLIPModel,
    labels: List[str] = ["an indoor scene", "an outdoor scene"],
) -> str:

    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image  
        idx = logits.argmax().item()

    return labels[idx]

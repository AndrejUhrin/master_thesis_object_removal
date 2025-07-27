import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection


def load_owl():
    model_name: str = "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name)
    return processor, model


def detect_with_owl(
    image,
    text_labels,
    processor,
    model,
    threshold: float = 0.13
):
    
    inputs = processor(text=text_labels, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)])

    result = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold,
        text_labels=text_labels,
    )
    result = result[0]
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["text_labels"]

    return boxes, scores, labels

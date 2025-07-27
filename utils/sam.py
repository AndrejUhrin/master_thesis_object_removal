import torch
from transformers import SamProcessor, SamModel

def load_sam():
    sam_model_id = "facebook/sam-vit-base"
    processor_sam = SamProcessor.from_pretrained(sam_model_id)
    model_sam = SamModel.from_pretrained(sam_model_id)
    return processor_sam, model_sam


def segment_with_sam(image, boxes, processor_sam, model_sam):
    if hasattr(boxes, "tolist"):
        boxes_list = boxes.tolist()
    else:
        boxes_list = boxes

    inputs_sam = processor_sam(image, input_boxes=[boxes_list], return_tensors="pt")

    with torch.no_grad():
        outputs_sam = model_sam(**inputs_sam)

    return outputs_sam, inputs_sam


def post_process_sam_masks(outputs_sam, processor_sam, inputs_sam):
    masks = processor_sam.image_processor.post_process_masks(
        outputs_sam.pred_masks.cpu(),
        inputs_sam["original_sizes"].cpu(),
        inputs_sam["reshaped_input_sizes"].cpu()
    )
    return masks




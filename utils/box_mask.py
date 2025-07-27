import torch
from PIL import Image, ImageOps, ImageDraw

def pad_to_square(img: Image.Image, fill: int = 114):
    w, h = img.size
    S = max(w, h)
    dw, dh = S - w, S - h
    left, top = dw//2, dh//2
    right, bottom = dw - left, dh - top
    padded = ImageOps.expand(img, border=(left, top, right, bottom),
                             fill=(fill, fill, fill))
    return padded, (left, top, right, bottom)

def unpad_boxes(boxes: list[list[float]], padding: tuple[int,int,int,int]):
    left, top, _, _ = padding
    return [
        [x0 - left, y0 - top, x1 - left, y1 - top]
        for x0, y0, x1, y1 in boxes
    ]

def enlarge_box(box: list[float], scale: float = 1.10,
               img_w: int = None, img_h: int = None):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + w/2, y0 + h/2
    nw, nh = w * scale, h * scale
    x0n, y0n = cx - nw/2, cy - nh/2
    x1n, y1n = cx + nw/2, cy + nh/2
    if img_w is not None and img_h is not None:
        x0n = max(0, min(img_w, x0n))
        x1n = max(0, min(img_w, x1n))
        y0n = max(0, min(img_h, y0n))
        y1n = max(0, min(img_h, y1n))
    return [x0n, y0n, x1n, y1n]

def draw_boxes_only(img: Image.Image, boxes: list[list[float]],
                    outline="red", width=3):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for x0,y0,x1,y1 in boxes:
        draw.rectangle([x0,y0,x1,y1], outline=outline, width=width)
    return out

def draw_annotated(img: Image.Image,
                   boxes: list[list[float]],
                   scores: list[float],
                   labels: list[str],
                   outline="red",
                   text_fill="red",
                   width=3,
                   font=None,
                   text_offset=(0, -10)):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for (x0,y0,x1,y1), s, lbl in zip(boxes, scores, labels):
        draw.rectangle([x0,y0,x1,y1], outline=outline, width=width)
        text = f"{lbl} ({s:.2f})"
        draw.text((x0 + text_offset[0], y0 + text_offset[1]),
                  text, fill=text_fill, font=font)
    return out

def create_box_mask(img_size: tuple[int,int],
                    boxes: list[list[float]]):
    """
    Return a grayscale (L) mask where box interiors are white (255).
    """
    w, h = img_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for x0,y0,x1,y1 in boxes:
        draw.rectangle([x0,y0,x1,y1], fill=255)
    return mask

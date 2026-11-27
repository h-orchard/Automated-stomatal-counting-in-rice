import base64
import io
from pathlib import Path
from PIL import Image, ImageDraw
import requests

MODEL_ID = "hannahs-workspace-qpks8/automated-stomatal-counting-400x-o2a7k-3-yolo11s-seg-t1"

def calculate_crop_area_mm2(
    image_width_px: int,
    image_height_px: int,
    image_width_mm: float
) -> float:
    """Calculate the physical area (mm²) of the defined crop region."""
    crop_width_px = (0.8176 - 0.2266) * image_width_px
    crop_height_px = 0.99445 * image_height_px
    mm_per_pixel = image_width_mm / image_width_px
    return crop_width_px * crop_height_px * (mm_per_pixel ** 2)

def infer_with_roboflow_api(crop: Image.Image, api_key: str, confidence: float) -> list:
    """Send cropped image directly to Roboflow Serverless HTTP API."""
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=95)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    response = requests.post(
        f"https://serverless.roboflow.com/{MODEL_ID}",
        params={
            "api_key": api_key,
            "confidence": confidence,
            "disable_preproc_static_crop": "true"
        },
        data=img_base64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30
    )
    response.raise_for_status()
    return response.json().get("predictions", [])

def process_single_image(
    api_key: str,
    image_bytes: bytes,
    filename: str,
    image_width_mm: float,
    confidence: float = 0.15,
    edge_margin: float = 0.5
) -> dict:
    """Crop a Dino-lite image, run YOLOv11 inference, filter edge boxes, and draw labels."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    # Crop to defined Dino-lite rectangle
    coordinates = (0.2266 * w, 0, 0.8176 * w, 0.99445 * h)
    crop = image.crop(coordinates)
    crop_width, crop_height = crop.size

    predictions = infer_with_roboflow_api(crop, api_key, confidence)

    # Exclude stomata touching the crop edges
    whole_stomates = []
    for pred in predictions:
        x, y = pred["x"], pred["y"]
        box_w, box_h = pred["width"], pred["height"]

        left = x - box_w / 2
        right = x + box_w / 2
        top = y - box_h / 2
        bottom = y + box_h / 2

        if (
            left > edge_margin
            and right < crop_width - edge_margin
            and top > edge_margin
            and bottom < crop_height - edge_margin
        ):
            whole_stomates.append(pred)

    # Draw bounding boxes on a copy of the cropped image
    annotated = crop.copy()
    draw = ImageDraw.Draw(annotated)
    for pred in whole_stomates:
        x, y = pred["x"], pred["y"]
        box_w, box_h = pred["width"], pred["height"]
        draw.rectangle(
            (x - box_w / 2, y - box_h / 2, x + box_w / 2, y + box_h / 2),
            outline="magenta",
            width=3
        )

    area_mm2 = calculate_crop_area_mm2(w, h, image_width_mm)
    stomate_count = len(whole_stomates)
    density = stomate_count / area_mm2 if area_mm2 > 0 else 0.0

    img_buffer = io.BytesIO()
    annotated.save(img_buffer, format="PNG")

    return {
        "image": filename,
        "stomate_count": stomate_count,
        "crop_area_mm2": round(area_mm2, 6),
        "stomatal_density_per_mm2": round(density, 2),
        "annotated_image": annotated,
        "annotated_bytes": img_buffer.getvalue(),
        "labelled_filename": f"{Path(filename).stem}.png"
    }

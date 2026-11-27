import io
from pathlib import Path
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


# Define models based on plant species
MODELS = {
    "Rice": "hannahs-workspace-qpks8/automated-stomatal-counting-400x-o2a7k-3-yolo11s-seg-t1",
    "Wheat": "field-dino-training-set-400x-guard-cell-only/3"
}

# Define fractional crop coordinates (left, top, right, bottom)
LEFT = 0.2266
RIGHT = 0.8176
TOP = 0
BOTTOM = 0.99445


def count_stomata(uploaded, api_key, image_width_mm, plant_type="Rice", confidence=0.15, edge_margin=1):
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    config = InferenceConfiguration(
        # set confidence threshol for model: lower confidence -> more flase positive, higher confidence -> more missed stomates
        confidence_threshold=confidence,
        # the initial model includes a crop which interferes with the later (better) crop so disable it
        disable_preproc_static_crop=True
    )

    # start an empty list for the results
    rows = []

    for filename, image_data in uploaded.items():
        # open the image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        w, h = image.size

        # only crop rice images; leave wheat images full-size
        if plant_type == "Rice":
            crop = image.crop((LEFT * w, TOP * h, RIGHT * w, BOTTOM * h))
            area_mm2 = (RIGHT - LEFT) * (BOTTOM - TOP) * (h / w) * (image_width_mm ** 2)
        else:
            crop = image
            area_mm2 = (h / w) * (image_width_mm ** 2)

        # run the model on the cropped image
        with CLIENT.use_configuration(config):
            predictions = CLIENT.infer(crop, model_id=ricestomata400x)["predictions"]

        ## only keep stomates that are fully in the image - exclude stomates whose outer boxes touch the edge
        crop_width, crop_height = crop.size
        whole_stomates = []

        for prediction in predictions:
            x, y = prediction["x"], prediction["y"]
            box_width, box_height = prediction["width"], prediction["height"]

            if (
                x - box_width / 2 > edge_margin
                and x + box_width / 2 < crop_width - edge_margin
                and y - box_height / 2 > edge_margin
                and y + box_height / 2 < crop_height - edge_margin
            ):
                whole_stomates.append(prediction)

        # only count stomates
        predictions = whole_stomates

        # draw a box around each counted stomate
        draw = ImageDraw.Draw(crop)
        for prediction in predictions:
            x, y = prediction["x"], prediction["y"]
            width, height = prediction["width"], prediction["height"]
            draw.rectangle(
                (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
                outline="magenta",
                width=3
            )

        # save the labelled image to memory
        img_buffer = io.BytesIO()
        crop.save(img_buffer, format="PNG")

        # count the stomata and calculate density
        stomate_count = len(predictions)
        area_mm2 = (RIGHT - LEFT) * (BOTTOM - TOP) * (h / w) * (image_width_mm ** 2)
        density = stomate_count / area_mm2

        # add the image name and count to the results
        rows.append({
            "image": filename,
            "stomate_count": stomate_count,
            "crop_area_mm2": round(area_mm2, 6),
            "stomatal_density_per_mm2": round(density, 2),
            "annotated_image": crop,
            "annotated_bytes": img_buffer.getvalue(),
            "labelled_filename": f"{Path(filename).stem}.png"
        })

    return rows


# Bridge so app.py can call process_single_image()
def process_single_image(api_key, image_bytes, filename, image_width_mm, plant_type="Rice", confidence=0.15, edge_margin=1):
    return count_stomata({filename: image_bytes}, api_key, image_width_mm, plant_type, confidence, edge_margin)[0]

import runpod

import cv2
import boto3
from PIL import Image
from io import BytesIO

import torch
import numpy as np
from ultralytics import YOLO

# globals
model = None
s3_client = None


assert (
    torch.cuda.is_available()
), "CUDA is not available. Make sure you have a GPU instance."


def load_model():
    model_path = "/runpod-volume/models/ultralytics/yolo11x-seg.pt"
    model = YOLO(model_path)
    model.to("cuda")
    return model


def handler(event):
    global model, s3_client

    # Download original image
    image_id = event["input"].get("id")
    if not image_id:
        return {"error": "No id provided for image segmentation."}

    image_key = f"img/{image_id}/original.png"

    try:
        # Initialize S3 client if not already created
        if s3_client is None:
            s3_client = boto3.client("s3")

        response = s3_client.get_object(Bucket="hasura-storage", Key=image_key)
        image_obj = BytesIO(response["Body"].read())

    except Exception as e:
        return {"error": f"Failed to process S3 operation: {str(e)}"}

    # Read original image and get its compression settings
    image = Image.open(image_obj)
    original_info = image.info

    # Initialize model if not already loaded
    if model is None:
        model = load_model()

    # Run inference with following settings:
    # - save=True      -> save the analytics result to disk
    # - save_crop=True -> save the cropped images to disk
    # - conf=0.25      -> set the confidence threshold to 25%
    results = model.predict(image, conf=0.25)

    # Create a BytesIO buffer for the annotated image
    annotated_buffer = BytesIO()

    # Plot and save the annotated image to the buffer
    result_plot = results[0].plot()  # returns a numpy array of the plotted image
    Image.fromarray(result_plot).save(
        annotated_buffer,
        format="PNG",
        **{
            k: v
            for k, v in original_info.items()
            if k in ["optimize", "compression", "compress_level"]
        },
    )
    annotated_buffer.seek(0)

    # Upload annotated image to S3
    annotated_key = f"img/{image_id}/segment.png"
    s3_client.upload_fileobj(
        Fileobj=annotated_buffer,
        Bucket="hasura-storage",
        Key=annotated_key,
        ExtraArgs={"ContentType": "image/png"},
    )

    # Process individual segments
    processed_objects = []
    for result in results:
        img = np.copy(result.orig_img)

        # Process each detected object
        for ci, contour_obj in enumerate(result):
            # Create binary mask for this object
            b_mask = np.zeros(img.shape[:2], np.uint8)

            # Get detection class name
            label = contour_obj.names[contour_obj.boxes.cls.tolist()[0]]

            # Extract and process contour
            contour = contour_obj.masks.xy[0]
            contour = contour.astype(np.int32)
            contour = contour.reshape(-1, 1, 2)

            # Draw contour onto mask
            cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

            # Save mask to S3
            mask_buffer = BytesIO()
            Image.fromarray(b_mask).save(mask_buffer, format="PNG")
            mask_buffer.seek(0)

            mask_key = f"img/{image_id}/masks/{label.replace(' ', '_')}_{ci}.png"
            s3_client.upload_fileobj(
                Fileobj=mask_buffer,
                Bucket="hasura-storage",
                Key=mask_key,
                ExtraArgs={"ContentType": "image/png"},
            )

            # Create isolated object with transparent background
            isolated = np.dstack([img, b_mask])

            # Get bounding box coordinates
            x1, y1, x2, y2 = (
                contour_obj.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            )

            # Crop image to object region
            iso_crop = isolated[y1:y2, x1:x2]

            # Convert numpy array to PIL Image
            image = Image.fromarray(iso_crop)

            # Save to BytesIO buffer
            buffered = BytesIO()
            image.save(
                buffered,
                format="PNG",
                **{
                    k: v
                    for k, v in original_info.items()
                    if k in ["optimize", "compression", "compress_level"]
                },  # Copy original compression settings
            )
            buffered.seek(0)

            # Upload to S3
            segment_key = f"img/{image_id}/segments/{label.replace(' ', '_')}_{ci}.png"
            s3_client.upload_fileobj(
                Fileobj=buffered,
                Bucket="hasura-storage",
                Key=segment_key,
                ExtraArgs={"ContentType": "image/png"},
            )

            processed_objects.append(
                {
                    "label": label,
                    "confidence": float(contour_obj.boxes.conf[0]),
                    "objectKey": segment_key,
                    "maskKey": mask_key,
                }
            )

    return {
        "key": annotated_key,
        "segments": processed_objects,
    }


runpod.serverless.start({"handler": handler})

import runpod

import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

import boto3
from PIL import Image
from io import BytesIO

# globals
model = None
s3_client = None

assert (
    torch.cuda.is_available()
), "CUDA is not available. Make sure you have a GPU instance."


def load_model(model_id):
    model = AutoModelForImageSegmentation.from_pretrained(
        model_id, trust_remote_code=True
    )
    torch.set_float32_matmul_precision(["high", "highest"][0])
    model.to("cuda")
    model.eval()
    return model


def handler(event):
    global model, s3_client

    # Download original image
    image_id = event["input"].get("id")
    if not image_id:
        return {"error": "No id provided for background removal."}

    image_key = f"img/{image_id}/original.png"

    # Retrieve image and read into memory
    try:
        # Initialize S3 client if not already created
        if s3_client is None:
            s3_client = boto3.client("s3")

        # Download from S3 and get image metadata
        response = s3_client.get_object(Bucket="your-bucket-name", Key=image_key)
        image_obj = BytesIO(response["Body"].read())
        image = Image.open(image_obj)
        original_info = image.info
    except Exception as e:
        return {"error": f"Failed to process S3 operation: {str(e)}"}

    # Initialize model if not already loaded
    model_id = event["input"].get("model", "briaai/RMBG-2.0")
    if model is None:
        model = load_model(model_id)
    print(f"Loaded model: {model_id}")

    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Predict
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    print("Mask prediction complete")

    # Resize mask to match original image size
    mask = pred_pil.resize(image.size)

    # Apply mask to image
    image.putalpha(mask)
    print("Mask applied to image")

    # Save image with transparent background to S3
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
    transparent_key = f"img/{image_id}/trans.png"
    s3_client.upload_fileobj(
        Fileobj=buffered,
        Bucket="your-bucket-name",
        Key=transparent_key,
        ExtraArgs={"ContentType": "image/png"},
    )
    print(f"Image saved to S3 with key: {transparent_key}")

    return {"key": transparent_key}


runpod.serverless.start({"handler": handler})

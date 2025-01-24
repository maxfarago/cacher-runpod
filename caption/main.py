import runpod

import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

import boto3
from PIL import Image
from io import BytesIO

# globals
model = None
processor = None
s3_client = None

assert (
    torch.cuda.is_available()
), "CUDA is not available. Make sure you have a GPU instance."


def load_model():
    global model, processor

    model_path = os.environ.get("MODEL_PATH", "/workspace/models/joycaption")
    model_id = "fancyfeast/llama-joycaption-alpha-two-hf-llava"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=0
    )
    # Save to network volume for future use
    print(f"Saving model to {model_path}")
    processor.save_pretrained(model_path)
    model.save_pretrained(model_path)
    model.eval()
    return processor, model


def handler(event):
    global model, processor, s3_client

    # Input validation
    image_id = event["input"].get("id")
    prompt = event["input"].get(
        "prompt", "Write a long descriptive caption for this image in a formal tone."
    )
    if not image_id:
        return {"error": "Missing required input parameters: id"}

    # Get requested image from S3
    image_key = f"img/{image_id}/original.png"
    try:
        # Reuse existing S3 client if available
        if s3_client is None:
            s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket="your-bucket-name", Key=image_key)
        image_obj = BytesIO(response["Body"].read())
        image = Image.open(image_obj)
    except Exception as e:
        return {"error": f"Failed to process S3 operation: {str(e)}"}

    # Initialize model if not already loaded
    if model is None or processor is None:
        processor, model = load_model()

    with torch.no_grad():
        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # Format the conversation
        convo_string = processor.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(convo_string, str)

        # Process the inputs
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(
            "cuda"
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # Generate the captions
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
            temperature=0.6,
            top_k=None,
            top_p=0.9,
        )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs["input_ids"].shape[1] :]

        # Decode the caption
        caption = processor.tokenizer.decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        caption = caption.strip()

        # Save the caption to S3
        try:
            caption_key = f"img/{image_id}/caption.txt"
            s3_client.put_object(
                Bucket="your-bucket-name", Key=caption_key, Body=caption
            )
        except Exception as e:
            return {"error": f"Failed to process S3 operation: {str(e)}"}

        return {
            "key": caption_key,
            "caption": caption,
        }


runpod.serverless.start({"handler": handler})

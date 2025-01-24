# Hasura Background Removal Service

A serverless GPU-powered background removal service built with RunPod, PyTorch, and the [RMBG-2.0 model](https://huggingface.co/briaai/RMBG-2.0) (developed on [BiRefNet architecture](https://github.com/ZhengPeng7/BiRefNet)).

## Overview

This service provides an API endpoint for removing backgrounds from images using state-of-the-art AI models. It processes images stored in S3 and returns transparent PNG files with the background removed.

## Features

- Serverless GPU inference using RunPod
- Support for multiple image formats (JPG, JPEG, PNG, GIF, WEBP)
- Automatic S3 integration for input/output
- Configurable model selection (defaults to RMBG-2.0)
- CUDA-accelerated processing
- Automatic image resizing and normalization
- Transparent PNG output with alpha channel

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker
- AWS S3 bucket access

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Setup

The service requires the following environment variables:

- AWS credentials configured for S3 access

## Docker Deployment

The service includes a Dockerfile for containerized deployment:

```bash
docker build -t mfarago/runpod-remove-bg --platform linux/amd64 .
docker run --gpus all background-removal-service
```

## Usage

### API Endpoint

The service expects a POST request with the following JSON payload:

```json
{
  "input": {
    "id": "your-image-id",
    "model": "briaai/RMBG-2.0" // optional, defaults to RMBG-2.0
  }
}
```

### Input/Output

- Input: Images should be stored in S3 under the `ogs/{image_id}` prefix
- Output: Processed images are stored as transparent PNGs at `img/{image_id}/trans.png`

## Technical Details

- Image Processing:

  - Input images are resized to 1024x1024
  - Normalization using ImageNet mean and std values
  - GPU-accelerated inference
  - Original aspect ratio preserved in output

- Model:
  - Default: RMBG-2.0
  - Custom models supported through input parameter
  - CUDA optimization enabled

## Response Format

Successful response:

```json
{
  "key": "img/{image_id}/trans.png"
}
```

Error response:

```json
{
  "error": "Error message description"
}
```

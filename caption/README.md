# Image Captioning Service

A serverless GPU-powered image captioning service built with RunPod, PyTorch, and the JoyCaption model.

## Overview

This service provides an API endpoint for captioning images using state-of-the-art AI models. It processes images stored in S3 and returns captions.

## Features

- Serverless GPU inference using RunPod
- Support for multiple image formats (JPG, JPEG, PNG, GIF, WEBP)
- Automatic S3 integration for input/output
- Configurable model selection (defaults to JoyCaption)
- CUDA-accelerated processing

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
docker build -t mfarago/runpod-caption --platform linux/amd64 .
docker run --gpus all runpod-caption
```

## Usage

### API Endpoint

The service expects a POST request with the following JSON payload:

```json
{
  "input": {
    "id": "your-image-id", // required
    "model": "fancyfeast/llama-joycaption-alpha-two-hf-llava" // optional
  }
}
```

### Input/Output

- Input: Images should be stored in S3 with the `img/{image_id}/original.png` key
- Output: Captions are stored as text at `img/{image_id}/caption.txt`

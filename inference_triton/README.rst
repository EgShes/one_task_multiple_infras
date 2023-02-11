# Nvidia Triton based inference

# Start

Build docker image

```bash
docker build --tag lpr_triton .
```

```bash
docker run \
    --gpus all \
    -v ./inference_triton/model_repository:/app/model_repository \
    -p 8000:8000 \
    --name lpr_triton \
    --rm \
    lpr_triton
```

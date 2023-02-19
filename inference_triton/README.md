# Nvidia Triton based inference

Recognition of russian car license plates using neural networks served by [Nvidia Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server).

# Start

```bash
cd docker
DOCKER_BUILDKIT=1 docker-compose -f prod.yml up --build
```

# Tests

1. Start Triton Inference Server

    See [Start](#start).

2. Install and activate virtual environment

    ```bash
    poetry install --with=dev
    poetry shell
    ```

3. Run tests

    ```bash
    pytest ./tests -vv --disable-warnings
    ```

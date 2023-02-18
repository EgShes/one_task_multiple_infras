# Nvidia Triton based inference

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
    poetry shell
    ```

3. Run tests

    ```bash
    pytest ./tests -vv
    ```

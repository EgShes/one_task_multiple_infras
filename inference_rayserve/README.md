# Ray Serve

Recognition of russian car license plates using neural networks served by [Ray Serve](https://www.ray.io/ray-serve).

## Run

Create virtual environment

```bash
python3.8 -m venv env
source env/bin/activate
pip install --upgrade pip pip-tools
pip-sync requirements.txt
```

### Locally

```bash
serve run inference_rayserve.service:app
```

### In Docker

```bash
DOCKER_BUILDKIT=1 docker build --tag inference_rayserve --build-context nn=../nn .
docker run --rm --name inference_rayserve -p 8000:8000 -p 8265:8265 --gpus all --shm-size=12g inference_rayserve
```

## Tests

### Unit

```bash
pytest -vv --disable-warnings tests/test_models.py
```

### Functional

```bash
pytest -vv --disable-warinings tests/test_service.py
```

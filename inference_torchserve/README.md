# TorchServe based inference

Recognition of russian car license plates using neural networks served by [TorchServe](https://pytorch.org/serve/).

## Start project

### Prod

### Dev

```bash
DOCKER_BUILDKIT=1 docker-compose -f common.yml -f dev.yml up --build --force-recreate
```

## Tests

1. Start project

2. Run tests

```bash
pytest -vv tests/functional
```

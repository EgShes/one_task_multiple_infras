# TorchServe based inference

Recognition of russian car license plates using neural networks served by [TorchServe](https://pytorch.org/serve/).

## Start project

### Prod

```bash
DOCKER_BUILDKIT=1 docker-compose -f common.yml -f prod.yml up --build --force-recreate
```

### Dev

```bash
DOCKER_BUILDKIT=1 docker-compose -f common.yml -f dev.yml up --build --force-recreate
```

## Tests

1. Start dev version of the project

2. Run tests

```bash
pytest -vv tests/functional
```

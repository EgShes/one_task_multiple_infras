# Prepare models for TorchServe

## Virtual environment

To install and activate venv run
```bash
poetry install --with=dev
poetry shell
```

## Yolo

```bash
torch-model-archiver --model-name yolo --version 1.0 --serialized-file ../nn/nn/weights/yolo.pt --handler inference_torchserve/handlers/yolo.py --export-path inference_torchserve/model_store/
```

## STN

```bash
python inference_torchserve/scripts/prepare_stn.py
torch-model-archiver --model-name stn --version 1.0 --serialized-file /tmp/stn.pt --handler inference_torchserve/handlers/stn.py --export-path inference_torchserve/model_store/
```

## LPRNET

```bash
python inference_torchserve/scripts/prepare_lprnet.py
torch-model-archiver --model-name lprnet --version 1.0 --serialized-file /tmp/lprnet.pt --handler inference_torchserve/handlers/lprnet.py --export-path inference_torchserve/model_store/
```

## Plate recognition

```bash
torch-workflow-archiver --workflow-name plate_recognition --spec-file inference_torchserve/workflow/plate_recognition.yaml --handler inference_torchserve/handlers/workflow/plate_recognition.py --export-path inference_torchserve/model_store/
```

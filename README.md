# Project goal

The goal of the project is to serve the same deep learning pipeline with different inference frameworks.

The task chosen is recognition of russian car license plates. Models were taken from [this repository](https://github.com/EtokonE/License_Plate_Recognition).

# Repository structure

- __nn__ - package with models
- __inference_triton__ - package for serving with [Nvidia Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)

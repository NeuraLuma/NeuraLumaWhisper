#!/bin/bash

# Install dependencies from yaml
source ~/.bash_profile
conda env create -f environmentOSX.yaml
conda activate NeuraLumaWhisper
# See Apple Instructions: https://developer.apple.com/metal/jax/
# Install JAX from source
git clone https://github.com/google/jax.git --branch jaxlib-v0.4.10 --single-branch
cd jax
python build/build.py --bazel_options=--@xla//xla/python:enable_tpu=true
python -m pip install dist/*.whl
python -m pip install jax==3.4.11
cd ..

# Install metal plugin
python -m pip install jax-metal

# Test
python -c 'import jax; jax.numpy.arange(10)'
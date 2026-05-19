#!/bin/bash
set -e

conda create -n HCPD python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate HCPD

conda install -y -c conda-forge libstdcxx-ng libgcc-ng pyarrow scikit-learn
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH

python -m pip install --upgrade pip setuptools wheel

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5.post1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]"

pip install transformers==4.52.3
pip install trl==0.18.0
pip install accelerate==1.4.0
pip install peft==0.16.0
pip install bleurt-pytorch
pip install seaborn

pip install morphcloud==0.1.67 --no-cache-dir
pip install liger_kernel==0.6.4
pip install mistral-common==1.8.6 --no-cache-dir
pip install email-validator==2.3.0

pip uninstall mergekit -y

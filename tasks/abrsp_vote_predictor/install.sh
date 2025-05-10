#! /bin/bash
set -e

git clone https://github.com/qinghezeng/ABRS-P /workspace/ABRS-P
cd /workspace/ABRS-P && git checkout 8ba3a7c

pip install pandas numpy torch torchmetrics scipy h5py wandb

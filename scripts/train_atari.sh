#!/bin/bash
set -ex
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
python ez/train.py exp_config=ez/config/exp/atari.yaml
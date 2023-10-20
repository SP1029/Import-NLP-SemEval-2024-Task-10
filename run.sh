#!/bin/bash

# Activate virtual environment

source /DATA2/venvs/pytorch_py39/bin/activate

cd /DATA2/ediref/Modules
/DATA2/venvs/pytorch_py39/bin/python train_erc_mmn.py


#!/bin/bash
rm -rf trial/checkpoints/*
python main.py --save_mesh --text "$1" --workspace trial -O

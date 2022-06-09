#!/bin/bash

python3 ./train.py --resume --epoch 40 --batch_size 64 --lr 0.0003 --weight_rot 0 --weight_ent 0.1 --data_root ../../datasets_dir/ROD-synROD/

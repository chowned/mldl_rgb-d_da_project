#!/bin/bash

epochs=40

#running code with default parameters
echo "Running train.py --epoch $epochs --batch_size 64"
python3 ./train.py --resume --epoch $epochs --batch_size 64 --data_root ../../datasets_dir/ROD-synROD/

#running code without DA
#echo "Running train.py --epoch $epochs --batch_size 64 --weight_rot 0"
#python3 ./train.py --resume --epoch $epochs --batch_size 64 --weight_rot 0 --data_root ../../datasets_dir/ROD-synROD/

#running code with higher learning rate
#echo "Running train.py --epoch $epochs --batch_size 64 --lr 0.0006 --weight_rot 1.0 --weight_ent 0.1"
#python3 ./train.py --resume --epoch $epochs --batch_size 64 --lr 0.0006 --data_root ../../datasets_dir/ROD-synROD/

#running code with higher weight entropy
#echo "Running train.py --resume --epoch $epochs --batch_size 64 --lr 0.0003 --weight_rot 1.0 --weight_ent 0.2"
#python3 ./train.py --resume --epoch $epochs --batch_size 64 --weight_rot 0 --weight_ent 0 --data_root ../../datasets_dir/ROD-synROD/

#!/bin/bash

epochs=20
batch_size=96


lr1=0.0003 # --lr
lr2=0.0006
lr_mult=1.1 # --lr_mult
weight_decay=0.04 # --weight_decay


echo "Careful, default batch size is 96, which needs 12GB of VRAM. Try 64 for 8GB or 32 for 4 GB"

#running code with default parameters
echo "1 Running train.py --epoch $epochs --batch_size $batch_size"
python3 ./train.py --resume --epoch $epochs --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/
clear

#running code without DA
echo "2 Running train.py --epoch $epochs --batch_size $batch_size --weight_rot 0 --weight_ent 0"
python3 ./train.py --resume --epoch $epochs --batch_size $batch_size --weight_rot 0 --weight_ent 0 --data_root ../../datasets_dir/ROD-synROD/
clear

#running code with higher lr
echo "3 Running train.py --epoch $epochs --batch_size $batch_size --lr $lr1"
python3 ./train.py --resume --epoch $epochs --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/ --lr $lr1
clear

#running code with higher lr
echo "4 Running train.py --epoch $epochs --batch_size $batch_size --lr $lr2"
python3 ./train.py --resume --epoch $epochs --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/ --lr $lr2
clear

#running code with higher lr_mult
echo "5 Running train.py --epoch $epochs --batch_size $batch_size --lr_mult $lr_mult"
python3 ./train.py --resume --epoch $epochs --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/ --lr_mult $lr_mult
clear

#running code with lower weight decay
echo "6 Running train.py --epoch $epochs --batch_size $batch_size --weight_decay $weight_decay"
python3 ./train.py --resume --epoch $epochs --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/ --weight_decay $weight_decay
clear

#!/bin/bash
clear

epochs=40
batch_size=64


lr1=0.0003 # --lr
lr2=0.0002
lr_mult=1.1 # --lr_mult
weight_decay=0.04 # --weight_decay


echo "Careful, default batch size is 96, which needs 12GB of VRAM. Try 64 for 8GB or 32 for 4 GB"
#printf "Please, insert new batch size: "
#read batch_size

#running code with lower weight decay
echo "1 Running train.py --epoch $epochs --batch_size $batch_size --weight_decay $weight_decay"
python3 ./train.py --resume --epoch $epochs --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/ --weight_decay $weight_decay
clear

#running code with lower weight decay and different implementation
echo "1 Running train_best_hp.py --epoch $epochs --batch_size $batch_size --weight_decay $weight_decay"
python3 ./train_best_hp.py --resume --epoch $epochs --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/ --weight_decay $weight_decay

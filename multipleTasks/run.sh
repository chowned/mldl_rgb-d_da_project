#!/bin/bash

epochs=40
epoch_test=10

#--weight_decay 0.05
#--lr_mult
#--lr

#running code with default parameters
echo "Running train.py --epoch $epochs --batch_size 64"
python3 ./train.py --resume --epoch $epochs --batch_size 64 --data_root ../../datasets_dir/ROD-synROD/

#running code without DA
#echo "Running train.py --epoch $epochs --batch_size 64 --weight_rot 0"
#python3 ./train.py --resume --epoch $epochs --batch_size 64 --weight_rot 0 --data_root ../../datasets_dir/ROD-synROD/

#running code with higher learning rate
echo "Running train.py --epoch $epoch_test --batch_size 64 --lr 0.0003"
python3 ./train.py --resume --epoch $epoch_test --batch_size 64 --lr 0.0003 --data_root ../../datasets_dir/ROD-synROD/

#running code with higher learning rate
echo "Running train.py --epoch $epoch_test --batch_size 64 --lr 0.0006 --weight_rot 1.0 --weight_ent 0.1"
python3 ./train.py --resume --epoch $epoch_test --batch_size 64 --lr 0.0006 --data_root ../../datasets_dir/ROD-synROD/

#running code with different lr and lr mult
echo "Running train.py --resume --epoch $epoch_test --batch_size 64 --lr 0.0003 --weight_rot 1.0 --weight_ent 0.2"
python3 ./train.py --resume --epoch $epoch_test --batch_size 64 --lr_mult 0.9 --lr 0.001 --data_root ../../datasets_dir/ROD-synROD/


#running code with default parameters
echo "Running train.py --epoch $epoch_test --batch_size 64 --weight_decay 0.04"
python3 ./train.py --resume --epoch $epoch_test --batch_size 64 --weight_decay 0.04 --data_root ../../datasets_dir/ROD-synROD/

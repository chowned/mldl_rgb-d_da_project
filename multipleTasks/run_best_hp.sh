#!/bin/bash
clear

epochs=20
batch_size=64


lr1=0.0003 # --lr
lr2=0.0002
lr_mult=1.1 # --lr_mult
weight_decay=0.04 # --weight_decay
weight_decay2=0.03 # --weight_decay
batch_size2=96


echo "Careful, default batch size is 96, which needs 12GB of VRAM. Try 64 for 8GB or 32 for 4 GB"
#printf "Please, insert new batch size: "
#read batch_size

#running code with normal weight decay
#echo "1 Running train.py --epoch $epochs --batch_size $batch_size "
#python3 ./train.py --resume --epoch $epochs --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/
#clear

#running code with lower weight decay
#echo "2 Running train.py --epoch $epochs --batch_size $batch_size --weight_decay $weight_decay"
#python3 ./train.py --resume --epoch $epochs --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/ --weight_decay $weight_decay
#clear

#running code with lower weight decay and different implementation
#echo "3 Running train_best_hp.py --epoch $epochs --batch_size $batch_size --weight_decay $weight_decay"
#python3 ./train_best_hp.py --resume --epoch $epochs --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/ --weight_decay $weight_decay

#running code with normal weight decay and different implementation
echo "4 Running train_best_hp.py --epoch $epochs --batch_size $batch_size --weight_decay $weight_decay"
python3 ./train_best_hp.py --resume --epoch $epochs --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/ --weight_decay $weight_decay

#running code with lower weight decay
#echo "5 Running train.py --epoch $epochs --batch_size $batch_size2 --weight_decay $weight_decay2"
#python3 ./train.py --resume --epoch $epochs --batch_size $batch_size2 --data_root ../../datasets_dir/ROD-synROD/ --weight_decay $weight_decay2
#clear

#running code with lower weight decay
#echo "6 Running train.py --epoch $epochs --batch_size $batch_size2 --weight_decay $weight_decay --lr $lr2"
#python3 ./train.py --resume --epoch $epochs --batch_size $batch_size2 --data_root ../../datasets_dir/ROD-synROD/ --weight_decay $weight_decay2 --lr $lr2
#clear


#running code with lower weight decay
echo "7 Running train_best_hp.py --epoch $epochs --batch_size $batch_size2 --weight_decay $weight_decay2"
python3 ./train_best_hp.py --resume --epoch $epochs --batch_size $batch_size2 --data_root ../../datasets_dir/ROD-synROD/ --weight_decay $weight_decay2 --lr $lr2
clear

#running code with lower weight decay
echo "8 Running train_best_hp.py --epoch $epochs --batch_size $batch_size2 --weight_decay $weight_decay2"
python3 ./train_best_hp.py --resume --epoch $epochs --batch_size $batch_size2 --data_root ../../datasets_dir/ROD-synROD/ --weight_decay $weight_decay2 
clear

#running code with lower weight decay
echo "9 Running train_best_hp.py --epoch $epochs --batch_size $batch_size2 --weight_decay $weight_decay2"
python3 ./train_best_hp.py --resume --epoch $epochs --batch_size $batch_size2 --data_root ../../datasets_dir/ROD-synROD/ --lr $lr2
clear
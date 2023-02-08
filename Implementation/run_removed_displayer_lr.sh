#!/bin/bash

#initial epoch
epochs=0 
#epoch steps
epoch_step=20
end_epoch=100
#other parameters
batch_size=96

lr=(0.003 0.0003 0.00003 0.000003 0.0000003)
lr_decay=0.1

lr1=0.0003 # --lr
lr2=0.0006
lr_mult=1.1 # --lr_mult
weight_decay=0.04 # --weight_decay

counter=0
end_epoch=100
end_counter=$((end_epoch/epoch_step))

echo "Careful, default batch size is 96, which needs 12GB of VRAM. Try 64 for 8GB or 32 for 4 GB"

for (( counter=0; counter<end_counter; counter++ ))
do  
    clear
    #lr=$( bc <<< "scale=10; $lr*$lr_decay")
    epochs=$((epochs+epoch_step))
    current_lr=${lr[counter]}
    echo "1 Running train_removed_lr_display.py --epoch $epochs --batch_size $batch_size --lr $current_lr"
    python3 ./train_removed_lr_display.py --resume --epoch $epochs --lr $current_lr --batch_size $batch_size --data_root ../../datasets_dir/ROD-synROD/
    
done


#!/bin/bash  

# python /mnt/HDD1/lc23/tyk/pgformer/src_difformer/main.py --id 61 --device 0 --output 3 --epochs 200 --batch_size 20000 \
#        --method difformer --lr 0.012 --num_layers 1 --num_heads 4    \
#               --input 128 --knn_num 1 --hidden_channels 256 --dropout 0.5 --weight_decay 1e-5 --alpha 0.5 --use_bn --use_residual --use_graph --save_model

# python /mnt/HDD1/lc23/tyk/pgformer/src_difformer/main.py --id 62 --device 0 --output 3 --epochs 200 --batch_size 20000 \
#        --method difformer --lr 0.012 --num_layers 1 --num_heads 4    \
#               --input 128 --knn_num 3 --hidden_channels 256 --dropout 0.5 --weight_decay 1e-5 --alpha 0.5 --use_bn --use_residual --use_graph --save_model

# python /mnt/HDD1/lc23/tyk/pgformer/src_difformer/main.py --id 63 --device 0 --output 3 --epochs 200 --batch_size 20000 \
#        --method difformer --lr 0.012 --num_layers 1 --num_heads 4    \
            #   --input 128 --knn_num 4 --hidden_channels 256 --dropout 0.5 --weight_decay 1e-5 --alpha 0.5 --use_bn --use_residual --use_graph --save_model

python /mnt/HDD1/lc23/tyk/pgformer/src_difformer/main.py --id 66 --device 3 --output 3 --epochs 200 --batch_size 20000 \
       --method difformer --lr 0.012 --num_layers 1 --num_heads 4    \
              --input 128 --knn_num 0 --hidden_channels 256 --dropout 0.5 --weight_decay 1e-5 --alpha 0.5 --use_bn --use_residual --use_graph --save_model



python /mnt/HDD1/lc23/tyk/pgformer/src_difformer/main.py --id 64 --device 3 --output 3 --epochs 2000 --batch_size 20000 \
       --method difformer --lr 0.012 --num_layers 1 --num_heads 4    \
              --input 96 --knn_num 2 --hidden_channels 256 --dropout 0.5 --weight_decay 1e-5 --alpha 0.5 --use_bn --use_residual --use_graph --save_model

# python /mnt/HDD1/lc23/tyk/pgformer/src_difformer/main.py --id 65 --device 0 --output 3 --epochs 200 --batch_size 20000 \
#        --method difformer --lr 0.012 --num_layers 1 --num_heads 4    \
#               --input 128 --knn_num 2 --hidden_channels 256 --dropout 0.5 --weight_decay 1e-5 --alpha 0.5 --use_bn --use_residual --use_graph --save_model


# python /mnt/HDD1/lc23/tyk/pgformer/src_difformer/main.py --id 66 --device 0 --output 3 --epochs 200 --batch_size 100000 \
#        --method difformer --lr 0.012 --num_layers 1 --num_heads 4    \
#               --input 192 --knn_num 2 --hidden_channels 256 --dropout 0.5 --weight_decay 1e-5 --alpha 0.5 --use_bn --use_residual --use_graph --save_model



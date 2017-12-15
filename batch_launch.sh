#!/usr/bin/env bash
data_dir=./data/multi30k
out_dir=./out
python_script=./src/main.py

nlayer_choice=( 1 2 )
emb_size_choice=( 150 300 )
n_hid_choice=( 100 200 300 )
tf_p_choice=( 0.3 0.5 0.7 )
dropout_choice=( 0 0.3 0.5 0.7 )
model_choice=( "GRU" "LSTM" )


string="$(date) - Running on $(hostname) - $run_name"

echo $string
echo $string >> logs

run_no=0
for nlayer in "${nlayer_choice[@]}"
do 
    for emb_size in "${emb_size_choice[@]}"
    do 
        for n_hid in "${n_hid_choice[@]}"
        do
            for tf_p in "${tf_p_choice[@]}"
            do 
                for dropout in "${dropout_choice[@]}"
                do 
                    for model in "${model_choice[@]}"
                    do 
                        save_dir=$out_dir/$run_no/bin
                        mkdir -p $save_dir
                        CUDA_VISIBLE_DEVICES=$1 python $python_script \
                             --cuda \
                             --data $data_dir \
                             --save $save_dir \
                             --lang en-de \
                             --lr_patience 1 \
                             --verbose \
                             --log-interval 20 \
                             --nhid $n_hid \
                             --emb_size  $emb_size \
                             --batch_size 100 \
                             --epochs 30 \
                             --lr 0.01 \
                             --nlayers $nlayer \
                             --clip 1.0 \
                             --dropout $dropout \
                             --use_attention \
                             --teacher_force_prob $tf_p \
                             --img_conditioning 1 \
                             --model GRU
                        run_no=$((run_no+1))
                    done
                done
            done
        done
    done
done
 
                         


      
             

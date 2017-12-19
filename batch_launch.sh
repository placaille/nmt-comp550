#!/usr/bin/env bash
data_dir=./data/multi30k
out_dir=./out
python_script=./src/main.py

nlayer_choice=( 1 ) #2 )
emb_size_choice=( 250 300 400 ) #300 )
n_hid_choice=( 500 600 ) #200 300 )
tf_p_choice=( 0.2 0.3 ) #0.5 0.8 )
dropout_choice=( 0.5 ) #0.7 )
model_choice=( "LSTM" ) #"LSTM" )


string="$(date) - Running on $(hostname) batch"

echo $string

run_no=433 # same but activate image conditioning
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
			echo $run_no
			run_no=$((run_no+1))
                        save_dir=$out_dir/$run_no/bin
                       	mkdir -p $save_dir
			sbatch --gres=gpu:1 \
			         --cpus-per-task=1 \
			         --mem=9999M \
			         --output=$out_dir/$run_no/log.txt \
			         --time=0-1:39 \
			         launch.sh $save_dir \
			 		   $n_hid \
			 		   $emb_size \
			 		   $nlayer \
					   $model \
			 		   $dropout \
       			 	           $tf_p 
                    done
                done
            done
        done
    done
done
 
                         


      
             

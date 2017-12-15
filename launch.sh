#!/usr/bin/env bash
data_dir=./data/multi30k
out_dir=./out
python_script=./src/main.py
path_emb=./bin/pre-trained_emb_layer.bin

if [ -z "$1" ]; then
	run_name="temp_run"
else
	run_name=$1
fi

save_dir=$out_dir/$run_name/bin

mkdir -p $save_dir
string="$(date) - Running on $(hostname) - $run_name"

echo $string
echo $string >> logs
python $python_script \
	--cuda \
	--data $data_dir \
	--save $save_dir \
	--path_emb $path_emb \
	--lang en-fr \
	--lr_patience 1 \
	--verbose \
	--log-interval 20 \
	--nhid 400 \
	--emb_size 300 \
	--batch_size 64 \
	--epochs 30 \
	--lr 0.01 \
	--nlayers 2 \
	--bidirectional \
	--model LSTM \
	--clip 1.0 \
	--dropout 0.3 \
	--teacher_force_prob 0.3 \
	--use_word_emb

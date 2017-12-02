#!/usr/bin/env bash
data_dir=./data/multi30k
out_dir=./out
python_script=./src/main.py

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
	--lang en-fr \
	--verbose \
	--log-interval 20 \
	--nhid 50 \
	--batch_size 100 \
	--epochs 20 \
	--lr 0.001 \
	--reverse_src \
	--nlayers 1 \
	--dropout 0.5 \
	--teacher_force_prob 0.3 \
	--use_attention

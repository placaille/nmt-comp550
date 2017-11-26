#!/usr/bin/env bash
data_dir=./data/multi30k
out_dir=./out
python_script=./src/main.py

if [ -z "$1" ]; then
	run_name="temp_run"
else
	run_name=$1
fi

save_dir=$out_dir/$run_name

mkdir -p $save_dir/bin
string="$(date) - Running on $(hostname) - $run_name"

echo $string
echo $string >> logs
python $python_script \
	--cuda \
	--data $data_dir \
	--save $save_dir/bin/model.pt \
	--lang en-fr \
	--verbose \
	--batch_size 20 \
	--lr 0.005 \
	--use-attention

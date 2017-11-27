#!/usr/bin/env bash
data_dir=./data/multi30k
out_dir=./out
python_script=./src/main.py

if [ -z "$1" ]; then
	run_name="temp_run"
else
	run_name=$1
fi

path_to_model=$out_dir/$run_name/bin

mkdir -p $out_dir/$run_name/preds
string="$(date) - Running on $(hostname) - $run_name"

echo $string
echo $string >> logs
python $python_script \
	--cuda \
	--data $data_dir \
	--path_to_model $path_to_model \
	--lang en-fr \
	--verbose \
	--batch_size 100

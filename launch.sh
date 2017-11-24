#!/usr/bin/env bash
data_dir=./data/multi30k
save_dir=./out/bin
python_script=./src/main.py

if [ -z "$1" ]; then
	run_name="temp_run"
else
	run_name=$1
fi

mkdir -p $save_dir/$run_name
string="$(date) - Running on $(hostname) - $run_name"

echo $string
echo $string >> logs
python $python_script \
	--cuda \
	--data $data_dir \
	--save $save_dir/$run_name \
	--lang en-fr \
	--verbose \
	--batch_size 1

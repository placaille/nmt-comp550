#!/usr/bin/env bash
data_dir=./data/multi30k
out_dir=./out
python_script=./src/generate.py

if [ -z "$1" ]; then
	run_name="temp_run"
else
	run_name=$1
fi

path_to_model=$out_dir/$run_name/bin
path_to_src=$data_dir/en-fr/val.en
path_to_tgt=$data_dir/en-fr/val.fr

mkdir -p $out_dir/$run_name/pred
mkdir -p $out_dir/$run_name/gold
string="$(date) - Generating valid on $(hostname) - $run_name"

echo $string
echo $string >> logs

python $python_script \
	--cuda \
	--data_src $path_to_src \
	--data_tgt $path_to_tgt \
	--path_to_model $path_to_model \
	--log-interval 5 \
	--verbose \
	--batch_size 20 \
	--beam_size 5

echo Initiating scoring..

valid_gold_path=$out_dir/$run_name/gold/gold_val_en-fr.txt
valid_pred_path=$out_dir/$run_name/pred/pred_val_en-fr.txt

echo Valid BLEU score
perl multi-bleu.perl $valid_gold_path < $valid_pred_path
echo completed.

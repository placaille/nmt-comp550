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

mkdir -p $out_dir/$run_name/pred
mkdir -p $out_dir/$run_name/gold
string="$(date) - Generating text on $(hostname) - $run_name"

echo $string
echo $string >> logs

python $python_script \
	--cuda \
	--data $data_dir \
	--path_to_model $path_to_model \
	--lang en-fr \
	--verbose \
	--batch_size 20 \
	--beam_size 5

echo Initiating scoring..

multi_bleu_sript=./scoring_scripts/multi-bleu.perl

train_gold_path=$out_dir/$run_name/gold/gold_train_en-fr.txt
valid_gold_path=$out_dir/$run_name/gold/gold_valid_en-fr.txt
test_gold_path=$out_dir/$run_name/gold/gold_test_en-fr.txt

train_pred_path=$out_dir/$run_name/pred/pred_train_en-fr.txt
valid_pred_path=$out_dir/$run_name/pred/pred_valid_en-fr.txt
test_pred_path=$out_dir/$run_name/pred/pred_test_en-fr.txt

echo Train BLEU score
perl multi-bleu.perl $train_gold_path < $train_pred_path
echo Valid BLEU score
perl multi-bleu.perl $valid_gold_path < $valid_pred_path
echo Test BLEU score
perl multi-bleu.perl $test_gold_path < $test_pred_path
echo completed.

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

mkdir -p $out_dir/$run_name/preds
mkdir -p $out_dir/$run_name/gold
string="$(date) - Generating text on $(hostname) - $run_name"

echo $string
echo $string >> logs
echo Starting translation..

python $python_script \
	--cuda \
	--data $data_dir \
	--path_to_model $path_to_model \
	--lang en-fr \
	--verbose \
	--batch_size 100

echo Initiating scoring..

perl_tokenizer=./scoring_scripts/tokenizer.perl
multi_bleu_sript=./scoring_scripts/multi-bleu.perl

# perl $perl_tokenizer 
# perl multi-bleu.perl $val_data_path_tgt < $remt_out_path/$out_name > \
# 	$remt_out_path/$run_name"_valid_score.info"
# echo completed.

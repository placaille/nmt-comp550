#!/usr/bin/env bash

usage() {
	echo " Usage : $0 [-d {test2016, test2017, val}] [-m <name-of-model>]"
}

# Flags
model=""
data=""
while getopts m:d: flag; do
  case $flag in
	m) model=$OPTARG ;;
	d) data=$OPTARG ;;
	*) usage; exit;;
	?) usage; exit;;
  esac
done

# check model and data specified
if [ -z $model ]; then
  echo "Missing argument -m"
  usage; exit;
fi
if [ -z $data ]; then
  echo "Missing argument -d"
  usage; exit;
fi

# GENERATION SCRIPT
data_dir=./data/multi30k
out_dir=./out
python_script=./src/generate.py
bleu_script=./multi-bleu.perl

path_to_model=$out_dir/$model/bin
path_to_src=$data_dir/en-fr/$data.en
path_to_tgt=$data_dir/en-fr/$data.fr

mkdir -p $out_dir/$model/pred
mkdir -p $out_dir/$model/gold
string="$(date) - Generating $data on $(hostname) - $model"

echo $string
echo $string >> logs

python $python_script \
	--cuda \
	--data_src $path_to_src \
	--data_tgt $path_to_tgt \
	--path_to_model $path_to_model \
	--log-interval 5 \
	--verbose \
	--batch_size 25 \
	--beam_size 15

pred_file=$out_dir/$model/pred/pred_$data\_en-fr.txt
gold_file=$out_dir/$model/gold/gold_$data\_en-fr.txt
pred_file_nounk=$out_dir/$model/pred/pred_$data\_en-fr_nounk.txt
gold_file_nounk=$out_dir/$model/gold/gold_$data\_en-fr_nounk.txt

echo Scoring $data predictions \(keeping unkown\)..
perl $bleu_script $gold_file < $pred_file
echo Scoring $data predictions \(removed unkown\)..
perl $bleu_script $gold_file_nounk < $pred_file_nounk
perl -E "print '=' x 89"

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

# change this to the path to the word2vec model (turn it on with --use_word2vec)
path_word2vec=/data/milatmp1/lacaillp/models/word_embeddings/GoogleNews-vectors-negative300.bin.gz

mkdir -p $save_dir
string="$(date) - Running on $(hostname) - $run_name"

echo $string
echo $string >> logs
python $python_script \
	--cuda \
	--data $data_dir \
	--save $save_dir \
	--path_word_emb $path_word2vec \
	--lang en-fr \
	--lr_patience 1 \
	--verbose \
	--log-interval 20 \
	--nhid 300 \
	--emb_size 200 \
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

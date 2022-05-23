#! /bin/sh

for time in {1..1}
do
	python3 eval_squad_hgt.py --model_type bert \
		--graph_type dep \
		--model_name_or_path bert-base-cased \
	    --data_dir ./data_dep/ \
			--output_dir ./test1/ \
			--tensorboard_save_path ./runs/test\
			--train_file train-v2.0.json \
			--predict_file dev-v2.0.json \
			--save_steps 1\
			--logging_steps 1\
			--do_train \
			--do_eval \
			--num_train_epochs 1\
			--evaluate_during_training \
			--begin_evaluation_steps 0\
			--learning_rate 2e-5 \
			--per_gpu_train_batch_size 32\
			--per_gpu_eval_batch_size 32\
			--overwrite_output_dir \
			--version_2_with_negative \
			--max_seq_length 384 \
			--threads 10\

done

#! /bin/sh

for time in {1..1}
do
	python3 train_squad_hgt.py --model_type bert \
		--model_name_or_path bert-base-cased \
		--graph_type dep \
	     --data_dir ./data/ \
			--output_dir ~/saved/bert-base-based/hgt_dep/\
			--tensorboard_save_path ./runs/bert_hgt_dep\
			--train_file train-v2.0.json \
			--predict_file dev-v2.0.json \
			--save_steps 1500\
			--logging_steps 800\
			--do_train\
			--do_eval \
			--num_train_epochs 3\
			--evaluate_during_training \
			--begin_evaluation_steps 4500\
			--learning_rate 2e-5 \
			--per_gpu_train_batch_size 32\
			--per_gpu_eval_batch_size 32\
			--overwrite_output_dir \
			--version_2_with_negative \
			--max_seq_length 384 \
			--threads 10\
			--gpus 6\

done

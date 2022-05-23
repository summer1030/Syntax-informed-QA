# Syntax-informed QA

The code for the work "Syntax-informed Question Answering with Heterogeneous Graph Transformer".

The environment and dependent libraries can be checked in requirement.txt

## Do parsing

    ** Have provided the parsed results (con_parsed_dev.json and con_parsed_train.json).
    ** Can skip this step and forward to the next step.

    ./utils/ConstituencyParse.py for consistuency parsing
    ./utils/DependencyParse.py for dependency parsing


## Process constituency parsing results to get the graphs

    _______________________________________

    For building consistuency graphs:

    ## for training data
    python ConGraphBuilding.py --data_split train --save_path ../data/squad_files/constituency_graphs/

    ## for development data
    python ConGraphBuilding.py --data_split dev --save_path ../data/squad_files/constituency_graphs/

    the generated files are saved to `save_path'
    _______________________________________

    For building dependency graphs:

    ## for training data
    python DepGraphBuilding.py --data_split train --save_path ../data/squad_files/dependency_graphs/

    ## for development data
    python DepGraphBuilding.py --data_split dev --save_path ../data/squad_files/dependency_graphs/

    the generated files are saved to `save_path'

    _______________________________________


## Train the model (use single gpu)

    bash ./run_train.sh

    Following are the parameters needed to be set. It can be set in the script (run_train.sh),
    and the details are described in the main training file (train_squad_hgt.py).

    python3 train_squad_hgt.py --model_type bert \
                --model_name_or_path bert-base-cased \
                --graph_type con \
                --data_dir ./data/ \
                --output_dir ~/saved/bert-base-based/hgt_con/\
                --tensorboard_save_path ./runs/bert_hgt_con\
                --train_file train-v2.0.json \
                --predict_file dev-v2.0.json \
                --save_steps 1500\
                --logging_steps 1500\
                --do_train\
                --do_eval \
                --num_train_epochs 3\
                --evaluate_during_training \
                --begin_evaluation_steps 4500\
                --learning_rate 2e-5 \
                --per_gpu_train_batch_size 32 \
                --per_gpu_eval_batch_size 32 \
                --overwrite_output_dir \
                --version_2_with_negative \
                --max_seq_length 384 \
                --threads 50\

## Evaluate a saved checkpoint

  bash ./run_eval.sh
   
  Need to set --do_train to initialize the graph 
  
  python3 eval_squad_hgt.py --model_type bert \
               --graph_type dep \
               --model_name_or_path bert-base-cased \
               --data_dir ./data_dep/ \
               --output_dir ./test/ \
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
      
##  

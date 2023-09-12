#!/bin/bash

export TASK_NAME=pos
export DATASET_NAME=udpos


# bs=16
bs=8  # batch size is set smaller than the original version due to the GPU memory restriction.
epoch=5
psl=16 # prefix sequence length
lr=1e-2
dropout=0.1
seeds=(10 42 421 520 1218)
model=$1
GPU=$2

for seed in "${seeds[@]}"
do 
  CUDA_VISIBLE_DEVICES=${GPU} python3 run.py \
    --model_name_or_path $model \
    --task_name $TASK_NAME \
    --dataset_name $DATASET_NAME \
    --do_train \
    --do_eval \
    --do_predict \
    --lang en \
    --max_seq_length 128 \
    --per_device_train_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --pre_seq_len $psl \
    --output_dir checkpoints/$DATASET_NAME-${model}-${psl}-${lr}-${seed}/ \
    --overwrite_output_dir \
    --hidden_dropout_prob $dropout \
    --seed ${seed} \
    --save_total_limit  1 \
    --log_level error \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --prefix
done
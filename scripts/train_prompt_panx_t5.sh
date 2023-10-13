#!/bin/bash

REPO=$PWD
GPU=${1:-0}
MODEL=${2:-google/mt5-base}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/topro"}
PATTERN_ID=${5:-0}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='panx'
export CUDA_VISIBLE_DEVICES=$GPU
LANGS='en,af,ar,az,bg,bn,de,el,es,et,eu,fa,fi,fr,gu,he,hi,hu,id,it,ja,jv,ka,kk,ko,lt,ml,mr,ms,my,nl,pa,pl,pt,qu,ro,ru,sw,ta,te,th,tl,tr,uk,ur,vi,yo,zh'
EPOCHS=10
MAX_LENGTH=128
LR=3e-5
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
elif [ $MODEL == "google/mt5-base" ] || [ $MODEL == "google/mt5-large" ]; then
  MODEL_TYPE="mt5"
fi
SEEDS=(10 42 421 520 1218)
#SEEDS=(10)

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
  LR=3e-5
else
  BATCH_SIZE=24
  GRAD_ACC=4
  #LR=1e-5
fi

runfewshot(){
  NAME="${MODEL}-LR${LR}-epoch${EPOCHS}-MaxLen${MAXL}-Pattern${PATTERN_ID}-seed${1}"
  SAVE_DIR="$OUT_DIR/$TASK/$PATTERN_ID/${NAME}/"
  RESULT_FILE="results_${TASK}_full.csv"
  mkdir -p $SAVE_DIR
  python $PWD/run_baseline/run_prompt_tag_t5.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --task_name $TASK \
    --do_train \
    --do_predict \
    --data_dir $DATA_DIR/${TASK} \
    --gradient_accumulation_steps $GRAD_ACC \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --per_gpu_eval_batch_size $BATCH_SIZE \
    --save_steps 1000 \
    --learning_rate $LR \
    --num_beam 1 \
    --num_train_epochs $EPOCHS \
    --max_seq_length $MAX_LENGTH \
    --output_dir $SAVE_DIR/ \
    --log_file 'train' \
    --predict_languages $LANGS \
    --save_only_best_checkpoint \
    --overwrite_output_dir \
    --eval_test_set $LC \
    --pattern_id $PATTERN_ID \
    --seed ${1} \
    --early_stopping
  #  --init_checkpoint outputs/xnli/bert-base-multilingual-cased-LR2e-5-epoch5-MaxLen128-PatternID1/checkpoint-best/
	
  # python $PWD/results_to_csv.py \
  #   --input_path "${SAVE_DIR}test_results.txt" \
  #   --save_path $RESULT_FILE \
  #   --name $NAME
}

for SEED in "${SEEDS[@]}"
do
  runfewshot $SEED
done

#--overwrite_cache \


#!/bin/bash

REPO=$PWD
GPU=${1:-1,2,3,4,5,6,7,0}
MODEL=${2:-bert-base-multilingual-cased}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}
PATTERN_ID=${5:-0}
MODEL_TYPE=${6:-bert}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='udpos'
export CUDA_VISIBLE_DEVICES=$GPU
LANGS='en,af,ar,bg,de,el,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,lt,mr,nl,pl,pt,ro,ru,ta,te,th,tl,tr,uk,ur,vi,wo,yo,zh'
EPOCHS=5
MAX_LENGTH=128
LR=1e-5
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi
SEEDS=(10 42 421 520 1218)
#SEEDS=(42)

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
  LR=3e-5
else
  BATCH_SIZE=8
  GRAD_ACC=4
  #LR=1e-5
fi

runfewshot(){
  NAME="${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-Pattern${PATTERN_ID}-seed${1}"
  SAVE_DIR="$OUT_DIR/$TASK/$PATTERN_ID/${NAME}/"
  RESULT_FILE="results_${TASK}_full.csv"
  mkdir -p $SAVE_DIR
  python $PWD/run_baseline/run_prompt_classify.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --task_name $TASK \
    --do_train \
    --do_predict \
    --data_dir $DATA_DIR/${TASK} \
    --gradient_accumulation_steps $GRAD_ACC \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --save_steps 1000 \
    --learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --max_seq_length $MAX_LENGTH \
    --output_dir $SAVE_DIR/ \
    --log_file 'train' \
    --predict_languages $LANGS \
    --save_only_best_checkpoint \
    --overwrite_output_dir \
    --overwrite_cache \
    --eval_test_set $LC \
    --pattern_id $PATTERN_ID \
    --seed ${1} \
    --early_stopping
  #  --init_checkpoint outputs/xnli/bert-base-multilingual-cased-LR2e-5-epoch5-MaxLen128-PatternID1/checkpoint-best/
	
  python $PWD/results_to_csv.py \
    --input_path "${SAVE_DIR}test_results.txt" \
    --save_path $RESULT_FILE \
    --name $NAME
}

for SEED in "${SEEDS[@]}"
do
  runfewshot $SEED
done

#!/bin/bash

REPO=$PWD
MODEL=${1:-google/mt5-base}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

TASK='udpos'
export CUDA_VISIBLE_DEVICES=$GPU
LANGS='en,af,ar,az,bg,bn,de,el,es,et,eu,fa,fi,fr,gu,he,hi,hu,id,it,ja,jv,ka,kk,ko,lt,ml,mr,ms,my,nl,pa,pl,pt,qu,ro,ru,sw,ta,te,th,tl,tr,uk,ur,vi,yo,zh'
NUM_EPOCHS=10
MAX_LENGTH=128
LR=3e-5
SEEDS=(10 42 421 520 1218)

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
elif [ $MODEL == "t5-base" ] || [ $MODEL == "google/mt5-base" ]; then
  MODEL_TYPE="t5"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=24
  GRAD_ACC=4
fi

OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}/"

run(){
    DATA_DIR1=$DATA_DIR/$TASK
    mkdir -p $OUTPUT_DIR
    python3 $REPO/run_baseline/run_tag_t5.py \
      --data_dir $DATA_DIR1 \
      --model_type $MODEL_TYPE \
      --task $TASK \
      --labels panx_labels.txt \
      --model_name_or_path $MODEL \
      --output_dir $OUTPUT_DIR \
      --max_seq_length  $MAX_LENGTH \
      --num_train_epochs $NUM_EPOCHS \
      --gradient_accumulation_steps $GRAD_ACC \
      --per_gpu_train_batch_size $BATCH_SIZE \
      --per_gpu_eval_batch_size 48 \
      --save_steps 1000 \
      --seed ${1} \
      --learning_rate $LR \
      --do_train \
      --do_eval \
      --do_predict \
      --predict_langs $LANGS \
      --log_file $OUTPUT_DIR/train.log \
      --eval_all_checkpoints \
      --overwrite_output_dir \
      --save_only_best_checkpoint $LC
}

for SEED in "${SEEDS[@]}"
do
  run $SEED
done
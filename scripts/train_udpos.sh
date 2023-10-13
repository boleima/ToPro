#!/bin/bash
# script modified based on:
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

TASK='udpos'
export CUDA_VISIBLE_DEVICES=$GPU
LANGS='en,af,ar,bg,de,el,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,lt,mr,nl,pl,pt,ro,ru,ta,te,th,tl,tr,uk,ur,vi,wo,yo,zh'
NUM_EPOCHS=5
MAX_LENGTH=128
LR=1e-5
SEEDS=(10 42 421 520 1218)

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=8
  GRAD_ACC=4
fi

OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}/"

run(){
    DATA_DIR=$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAX_LENGTH}/
    RESULT_FILE="results_${TASK}_bl_full_${1}.csv"
    mkdir -p $OUTPUT_DIR
    python3 $REPO/run_baseline/run_tag.py \
      --data_dir $DATA_DIR \
      --model_type $MODEL_TYPE \
      --labels udpos_labels.txt \
      --model_name_or_path $MODEL \
      --output_dir $OUTPUT_DIR \
      --max_seq_length  $MAX_LENGTH \
      --num_train_epochs $NUM_EPOCHS \
      --gradient_accumulation_steps $GRAD_ACC \
      --per_gpu_train_batch_size $BATCH_SIZE \
      --save_steps 1000 \
      --seed ${1} \
      --learning_rate $LR \
      --do_train \
      --do_eval \
      --do_predict \
      --evaluate_during_training \
      --predict_langs $LANGS \
      --log_file $OUTPUT_DIR/train.log \
      --eval_all_checkpoints \
      --overwrite_output_dir \
      --save_only_best_checkpoint $LC
    python $PWD/results_to_csv2.py \
        --input_path "${OUTPUT_DIR}test_results.txt" \
        --save_path $RESULT_FILE \
        --name "${TASK}seed${1}" #$NAME
}

for SEED in "${SEEDS[@]}"
do
  run $SEED
done

#!/bin/bash

TASK=${1:-udpos}
MODEL=${2:-bert-base-multilingual-cased}
METHOD=${3:-topro}

if [ "$MODEL" == "mt5-base" ]
    then
        if [ "$METHOD" == "topro" ]
            then
                source scripts/train_prompt_"$TASK"_t5.sh 0 "google/mt5-base"
            else
                source scripts/train_"$TASK"_t5.sh "google/mt5-base" 0
        fi
    else
        if [ "$METHOD" == "topro" ]
            then
                source scripts/train_prompt_$TASK.sh 0 $MODEL
            else
                source scripts/train_$TASK.sh $MODEL 0
        fi
fi


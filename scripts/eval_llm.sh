#!/bin/bash

TASK=${1:-udpos}

source scripts/eval_prompt_llm_"$TASK"_direct.sh 0 "bigscience/bloomz-7b1" bloom
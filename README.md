## Overview

**ToPro: Token-Level Prompt Decomposition for Cross-Lingual Sequence Labeling Tasks**

Anonymous ARR submission

Last update: October 14, 2023

Abstract: 

Prompt-based methods have been successfully applied to multilingual pretrained language models for zero-shot cross-lingual understanding. However, most previous studies primarily focused on sentence-level classification tasks, and only a few considered token-level labeling tasks such as Named Entity Recognition (NER) and Part-of-Speech (POS) tagging. 

In this paper, we propose Token-Level Prompt Decomposition (ToPro) which facilitates the application of prompt-based learning to token-level sequence labeling tasks. Our ToPro method decomposes the input sentence into single tokens and applies one prompt template to each token. 

Our experiments on multilingual NER (PAN-X) and POS tagging (UDPOS) datasets demonstrate that ToPro-based fine-tuning outperforms vanilla fine-tuning and prompt tuning in cross-lingual transfer, especially for languages that are typologically different from the source language English. Our method also attains state-of-the-art performance when employed with mT5 model. Besides, our exploratory study in multilingual large language models shows that ToPro performs much better than the current in-context learning method. 

Overall, the performance improvement of ToPro shows that it could potentially be a novel benchmarking method for sequence labeling tasks.


## Content

- `run_baseline`: py scripts for Vanilla and ToPro. The code for the Prompt-Tuning (PT) baseline ([Tu et al. 2022](https://arxiv.org/pdf/2210.12360.pdf)) is available in their original repository: [https://github.com/salesforce/MPT](https://github.com/salesforce/MPT).
- `scripts`: sh scripts to run the models


## Dataset
The two datasets (UDPOS and PAN-X) in the current study are based on the XTREME benchmark ([Hu et al. 2020](https://arxiv.org/pdf/2003.11080.pdf)). The datasets can be accessed from this repository [https://github.com/google-research/xtreme](https://github.com/google-research/xtreme).


## Run
task: `udpos`, `panx`  
model: `bert-base-multilingual-cased`, `xlm-roberta-base`, `mt5-base`  
method: `topro`, `vanilla`, [PT](https://github.com/salesforce/MPT)  

- #### run ToPro and baseline training: 
```
source scripts/train.sh [task] [model] [method]
```

- #### run ToPro evaluation on LLM: 
```
source scripts/eval_llm.sh [task]
```



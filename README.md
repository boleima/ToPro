## Overview

Title: **TOPRO: Token-Level Prompt Decomposition for Cross-Lingual Sequence Labeling Tasks**

Last update: October 15, 2023

Prompt-based methods have been successfully applied to multilingual pretrained language models for zero-shot cross-lingual understanding. However, most previous studies primarily focused on sentence-level classification tasks, and only a few considered token-level labeling tasks such as Named Entity Recognition (NER) and Part-of-Speech (POS) tagging. 

In this paper, we propose Token-Level Prompt Decomposition (Topro) which facilitates the application of prompt-based learning to token-level sequence labeling tasks. Our method uses a prompt template to reformulate a token-level labeling task into a series of sequence labeling tasks -- one for each token. 

Our experiments on multilingual NER (PAN-X) and POS tagging (UDPOS) datasets demonstrate that Topro fine-tuning outperforms vanilla fine-tuning and prompt tuning in cross-lingual transfer, especially for languages that are typologically different from the source language English. Our method also attains state-of-the-art (SOTA) performance when employed with mT5 model. Besides, our exploratory study in multilingual large language models shows that Topro performs much better than the current in-context learning method. 

Overall, the performance improvement of Topro shows that it could be a potential novel benchmarking method for sequence labeling tasks.

## Dataset
The two datasets (UDPOS and PAN-X) of the current study are based on the XTREME benchmark ([Hu et al. 2020](https://arxiv.org/pdf/2003.11080.pdf)). The datasets can be accessed from this repository [https://github.com/google-research/xtreme](https://github.com/google-research/xtreme).


## Content

- `run_baseline`: py scripts for Vanilla and Topro. The code for the Prompt-Tuning (PT) baseline ([Tu et al. 2022](https://arxiv.org/pdf/2210.12360.pdf)) is available in their original repository: [https://github.com/salesforce/MPT](https://github.com/salesforce/MPT).
- `scripts`: sh scripts to run the models


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



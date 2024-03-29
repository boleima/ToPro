## Overview

**[ToPro: Token-Level Prompt Decomposition for Cross-Lingual Sequence Labeling Tasks](https://aclanthology.org/2024.eacl-long.164)** (EACL2024)

Abstract: 

Prompt-based methods have been successfully applied to multilingual pretrained language models for zero-shot cross-lingual understanding. However, most previous studies primarily focused on sentence-level classification tasks, and only a few considered token-level labeling tasks such as Named Entity Recognition (NER) and Part-of-Speech (POS) tagging. 

In this paper, we propose **To**ken-Level **Pro**mpt Decomposition (**ToPro**), which facilitates the prompt-based method for token-level sequence labeling tasks. The ToPro method decomposes an input sentence into single tokens and applies one prompt template to each token. 

Our experiments on multilingual NER and POS tagging datasets demonstrate that ToPro-based fine-tuning outperforms Vanilla fine-tuning and Prompt-Tuning in zero-shot cross-lingual transfer, especially for languages that are typologically different from the source language English. Our method also attains state-of-the-art performance when employed with the mT5 model. Besides, our exploratory study in multilingual large language models shows that ToPro performs much better than the current in-context learning method. 

Overall, the performance improvements show that ToPro could potentially serve as a novel and simple benchmarking method for sequence labeling tasks.


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

## Citation

If you found the resources in this repository useful, please cite:

```
@inproceedings{ma-etal-2024-topro,
    title = "{T}o{P}ro: Token-Level Prompt Decomposition for Cross-Lingual Sequence Labeling Tasks",
    author = {Ma, Bolei  and
              Nie, Ercong  and
              Yuan, Shuzhou  and
              Schmid, Helmut  and
              F{\"a}rber, Michael  and
              Kreuter, Frauke  and
              Schuetze, Hinrich},
    editor = "Graham, Yvette  and
              Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.164",
    pages = "2685--2702",
}
```

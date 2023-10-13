## Overview

Title: **TOPRO: Token-Level Prompt Decomposition for Cross-Lingual Sequence Labeling Tasks**

Last update: October 15, 2023

Prompt-based methods have been successfully applied to multilingual pretrained language models for zero-shot cross-lingual understanding. However, most previous studies primarily focused on sentence-level classification tasks, and only a few considered token-level labeling tasks such as Named Entity Recognition (NER) and Part-of-Speech (POS) tagging. 

In this paper, we propose Token-Level Prompt Decomposition (Topro) which facilitates the application of prompt-based learning to token-level sequence labeling tasks. Our method uses a prompt template to reformulate a token-level labeling task into a series of sequence labeling tasks -- one for each token. 

Our experiments on multilingual NER (PAN-X) and POS tagging (UDPOS) datasets demonstrate that Topro fine-tuning outperforms vanilla fine-tuning and prompt tuning in cross-lingual transfer, especially for languages that are typologically different from the source language English. Our method also attains state-of-the-art (SOTA) performance when employed with mT5 model. Besides, our exploratory study in multilingual large language models shows that Topro performs much better than the current in-context learning method. 

Overall, the performance improvement of Topro shows that it could be a potential novel benchmarking method for sequence labeling tasks.

## Content

- `run_baseline`: py scripts for baselines and Topro
- `scripts`: sh scripts to run the models



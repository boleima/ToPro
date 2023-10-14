# -*- coding: utf-8 -*-

import os
import pickle
import random
from typing import Dict, List

from sentence_transformers import SentenceTransformer, models

import logging
from .utils import InputExample, cosine_similarity

logger = logging.getLogger(__name__)

def get_sim_sents(original_sents_ex: List[InputExample], sent_pool, model: SentenceTransformer, task_name: str,
                  num_sim_sents: int, self_prediction: bool = False) \
        -> Dict[InputExample, List[InputExample]]:
    """
    Retrieve cross-lingual semantically sentences from a pool of sentences in high-resource language (e.g., English)
    for a given list of sentences in low-resource language.

    :param original_sents_ex: the list of sentences for which similar sentences should be retrieved
    :param sent_pool: the list of sentences from which similar sentences are retrieved
    :param model: the sentence transformer model used for sentence encoding
    :param num_sim_sents: the number of retrieved similar sentences for each original sentence
    :param self_prediction: wheather to use self-prediction method at cross-lingual retrieval
    :return: a dictionary mapping original sentence to its 100 most similar sentences in a foreign language
    """

    ori_sents = [e.text_a for e in original_sents_ex]
    embedded_ori_sents = model.encode(ori_sents)
    embedded_sent_pool = model.encode(sent_pool[0])

    sim_mat = cosine_similarity(embedded_ori_sents, embedded_sent_pool)

    # store the indices of k most similar sentences, k = num_sim_sents
    k_sim_sent_indices = list()
    for row in sim_mat:
        sim_sent_indices = row.argsort()[::-1][:num_sim_sents]
        k_sim_sent_indices.append(sim_sent_indices)

    ori_sent_to_sim_sents = dict()

    for (idx, sent) in enumerate(original_sents_ex):
        sent_indices = k_sim_sent_indices[idx]

        candidates = list()
        for c_idx in sent_indices:
            if self_prediction:
                if task_name == 'xnli':
                    text_a, text_b = sent_pool[c_idx].split('\t')
                    candidate = InputExample(guid=c_idx, text_a=text_a, text_b=text_b)
                else:
                    candidate = InputExample(guid=c_idx, text_a=sent_pool[c_idx])
            else:
                label = sent_pool[1][c_idx]
                # label = '1' if star < 3 else '2'
                if task_name == 'xnli':
                    text_a, text_b = sent_pool[0][c_idx].split('\t')
                    candidate = InputExample(guid=c_idx, text_a=text_a, text_b=text_b, label=label)
                else:
                    candidate = InputExample(guid=c_idx, text_a=sent_pool[0][c_idx], label=label)
            candidates.append(candidate)

        ori_sent_to_sim_sents[sent] = candidates

    return ori_sent_to_sim_sents

# get_sim_sents(eval_data, sent_pool, sent_encoder, num_sim_sent, self_prediction)
def get_random_sents(original_sents_ex: List[InputExample], sent_pool, num_sim_sents: int, seed=1213):
    sent_pool = list(zip(sent_pool[0], sent_pool[1]))
    random.seed(seed)
    random.shuffle(sent_pool)
    pool_size = len(sent_pool)

    ori_sent_to_sim_sents = dict()
    for example in original_sents_ex:
        rand_idx = random.randint(0, pool_size - num_sim_sents)
        priming_sents = sent_pool[rand_idx : rand_idx+num_sim_sents]
        candidates = []
        for idx, sent in enumerate(priming_sents):
            candidate = InputExample(guid=idx, text_a=sent[0], label=sent[1])
            candidates.append(candidate)

        ori_sent_to_sim_sents[example] = candidates

    return ori_sent_to_sim_sents

def save_sim_sents(sents: Dict[str, List[str]], save_path: str):
    with open(save_path, 'wb') as f:
        pickle.dump(sents, f)

def retrieve_sim_labeled_sents(eval_data: List[InputExample], data_dir: str, task_name: str,
                               transformer_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                               lang: str = 'en', num_sim_sent: int = 100, save: bool = False,
                               method: str = 'sentence_transformer', seed: int = 42,
                               num_priming: int=0, random_retrieval: bool = False,
                               ) -> Dict[InputExample, List[InputExample]]:
    """
    Retrieve the candidates from sentence pool most similar to the input sequence together with the label predicted by
    by the model.

    :param eval_data: list of evaluation data examples
    :param data_dir: the directory of the input sequence data
    :param transformer_name: the name of the sentence transformer used for sentence retrieval
    :param lang: the high-resource language of the sentence pool
    :param num_sim_sent: the number of the retrieved sentence
    :param save: if save the retrieved sentences or not
    :param method: which information retrieval method to use
    :param seed: random seed for initialization
    :param self_prediction: wheather to use self-prediction at cross-lingual retrieval
    :param num_priming: the number of retrieved cross-lingual priming sentences
    :param random_retrieval: execute random retrieval
    :return: a dictionary mapping a input sequence to its high-resource similar sentences with the label
    """

    sent_pool_file_path = os.path.join(data_dir, 'sent_pool_en.txt')
    sent_pool = []
    labels = []
    with open(sent_pool_file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if task_name == 'xnli':
                text_a, text_b, label = line.split('\t')
                sent = text_a+'\t'+text_b
            elif task_name == 'xtc':
                if line.startswith('id'):
                    continue
                else:
                    idx, _, label, sent = line.split('\t')
            else:
                try:
                    sent, label = line.split('\t')
                except:
                    continue
            sent_pool.append(sent.strip())
            labels.append(label.strip())

    sent_pool = (sent_pool, labels)

    logger.info('Load original data from file')

    # retrieve similar sentences by sentence transformer
    if random_retrieval:
        ori_sent_to_sim_sents = get_random_sents(eval_data, sent_pool, num_priming, seed)
        logger.info('create dictionary mapping original sentence to random sentences.')
    else:
        # load sentence transformer
        if transformer_name == 'average_pooling':
            # load sentence transformer by combining PLM with average pooling method
            mbert = models.Transformer('bert-base-multilingual-cased')
            emb_dim = mbert.get_word_embedding_dimension()
            pooling = models.Pooling(emb_dim)
            sent_encoder = SentenceTransformer(modules=[mbert, pooling])
        else:
            # load from pretrained sentence transformer
            sent_encoder = SentenceTransformer(transformer_name)

        ori_sent_to_sim_sents = get_sim_sents(eval_data, sent_pool, sent_encoder, task_name, num_sim_sent)
        logger.info(f'create dictionary mapping original sentence to similar sentences.')

        if save:
            save_file_name = f'sim_sents_{str(lang)}_method{str(method)}.pk'
            save_path = os.path.join(data_dir, save_file_name)
            save_sim_sents(ori_sent_to_sim_sents, save_path)

    return ori_sent_to_sim_sents


        # logger.info('Saving logits as pickle file.')
        # with open('tmp.pk', 'wb') as f:
        #     pickle.dump(labeled_ori_sent_to_sim_sents, f)

def add_priming_data(eval_data: List[InputExample], data_dir: str, task_name,
                     transformer_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                     lang: str = 'en', num_sim_sent: int = 100, save: bool = False, method: str = 'sentence_transformer',
                     seed: int = 42, num_priming: int=0, random_retrieval: bool = False) -> List[InputExample]:

    labeled_ori_sent_to_sim_sents = retrieve_sim_labeled_sents(eval_data=eval_data, data_dir=data_dir,
                                                               task_name=task_name, transformer_name=transformer_name,
                                                               lang=lang, num_sim_sent=num_sim_sent, save=save,
                                                               method=method, seed=seed, num_priming=num_priming,
                                                               random_retrieval=random_retrieval)


    for example in eval_data:
        example.meta = {'priming_data': labeled_ori_sent_to_sim_sents[example]}

    return eval_data
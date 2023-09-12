# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""on panx and udpos task, borrowed from (Finetuning multi-lingual models on XNLI/PAWSX (Bert, XLM, XLMRoberta).)"""

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForMaskedLM,
    get_linear_schedule_with_warmup,
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
)

from processors.xnli import XnliProcessor
from processors.pawsx import PawsxProcessor
from processors.amazon import AmazonProcessor
from processors.udpos import UdposProcessor
from processors.panx import PanxProcessor
from preprocessor import MLMPreprocessor
from processors.retrieve import add_priming_data

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ARCHIVES = (BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
            XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
            XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)

ALL_MODELS = sum(
    (tuple(archive_map.keys())
     for archive_map in ARCHIVES),
    ()
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "xlmr": (XLMRobertaConfig, XLMRobertaForMaskedLM, XLMRobertaTokenizer),
}

PROCESSORS = {
    'xnli': XnliProcessor,
    'pawsx': PawsxProcessor,
    'amazon': AmazonProcessor,
    'udpos':UdposProcessor,
    'panx':PanxProcessor
}




def compute_metrics(preds, labels):
    scores = {
        "f1": f1_score(preds,labels,average='weighted'),
        "num": len(preds),
        "correct": (preds == labels).sum(),
        "precision": precision_score(preds,labels,average='weighted'),
        "recall": recall_score(preds,labels,average='weighted'),
    }
    return scores


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, preprocessor, label_list, lang2id=None):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        if args.model_name_or_path.split('-')[-1].split('/')[0].isnumeric():
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    best_score = 0
    previous_score = 0
    best_checkpoint = None
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    num_worse_epoch = 0
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3],
                      "mlm_labels": batch[4], 'idx': batch[5]}
            # if args.model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if args.model_type in ["bert"] else None
            #     )  # XLM don't use segment_ids
            # if args.model_type == "xlm":
            #     inputs["langs"] = batch[4]
            # outputs = model(**inputs)
            # loss = outputs[0]
            loss = mlm_train_step(batch=inputs, preprocessor=preprocessor, model=model, label_list=label_list)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    # Only evaluate on single GPU otherwise metrics may not average well
                    if (args.local_rank == -1 and args.evaluate_during_training):
                        results = evaluate(args, model, preprocessor, split=args.train_split, language=args.train_language,
                                           lang2id=lang2id)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                #  use cross-lingual transfer results (average acc.) on the dev-set all languages to decide the best checkpoint  
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.eval_test_set:
                        output_predict_file = os.path.join(args.output_dir, 'eval_test_results')
                        total = total_correct = 0.0
                        with open(output_predict_file, 'a') as writer:
                            writer.write('\n======= Predict using the model from checkpoint-{}:\n'.format(global_step))
                            result = evaluate(args, model, preprocessor, split=args.dev_split, language=args.train_language,
                                                lang2id=lang2id, prefix='checkpoint-' + str(global_step))
                            avg_f1 = result['f1']
                            writer.write('{}={}\n'.format(args.train_language, result['f1']))

   
                    if args.save_only_best_checkpoint:
                        logger.info(" Dev f1 of train language = {}".format(avg_f1))
                        if avg_f1 > best_score:
                            logger.info(" average f1={} > best_score={}".format(avg_f1, best_score))
                            output_dir = os.path.join(args.output_dir, "checkpoint-best")
                            best_checkpoint = output_dir
                            best_score = avg_f1
                            # Save model checkpoint
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    if args.early_stopping:
                        if avg_f1 > previous_score:
                            num_worse_epoch = 0
                        else:
                            num_worse_epoch += 1
                        previous_score = avg_f1

                    else:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if (args.max_steps > 0 and global_step > args.max_steps) or (args.early_stopping and num_worse_epoch==args.tolerance):
                epoch_iterator.close()
                break
        if (args.max_steps > 0 and global_step > args.max_steps) or (args.early_stopping and num_worse_epoch==args.tolerance):
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    
    if global_step != 0:
        tr_loss = tr_loss / global_step

    return global_step, tr_loss, best_score, best_checkpoint


def evaluate(args, model, preprocessor, split='train', language='en', lang2id=None, prefix=""):
    """Evalute the model."""
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    total_results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        results = {}
        for idx in range(args.num_priming):
            eval_dataset = load_and_cache_examples(args, eval_task, preprocessor, split=split, language=language,
                                                   lang2id=lang2id, evaluate=True, priming_idx=idx)

            if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # multi-gpu eval
            model = model.to('cuda:0')
            # if args.n_gpu > 1:
            #     model = torch.nn.DataParallel(model)

            # Eval!
            logger.info("***** Running evaluation {} {} *****".format(prefix, language))
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            # eval_loss = 0.0
            # nb_eval_steps = 0
            preds = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3],
                          "mlm_labels": batch[4], 'idx': batch[5]}
                labels = inputs['labels']
                indices = inputs['idx']

                with torch.no_grad():
                    # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    # if args.model_type != "distilbert":
                    #     inputs["token_type_ids"] = (
                    #         batch[2] if args.model_type in ["bert"] else None
                    #     )  # XLM and DistilBERT don't use segment_ids
                    # if args.model_type == "xlm":
                    #     inputs["langs"] = batch[4]
                    # outputs = model(**inputs)
                    logits = mlm_eval_step(inputs, preprocessor, model)
                    # tmp_eval_loss, logits = outputs[:2]
                #
                #     eval_loss += tmp_eval_loss.mean().item()
                # nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                    all_indices = indices.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                    all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)

            # eval_loss = eval_loss / nb_eval_steps
            if args.output_mode == "classification":
                predictions = np.argmax(preds, axis=1)
            else:
                raise ValueError("No other `output_mode` for XNLI.")

            temp_results = {}
            temp_results['indices'] = all_indices
            temp_results['predictions'] = np.expand_dims(predictions, 0)
            temp_results['logits'] = preds
            temp_results['labels'] = out_label_ids

            if results:
                results['logits'] = np.concatenate((results['logits'], temp_results['logits']), axis=1)
                results['predictions'] = np.concatenate((results['predictions'], temp_results['predictions']), axis=0)
            else:
                results.update(temp_results)
            if args.aug_strategy == 'conc':
                break

        results['final_predictions'] = np.array([np.bincount(l).argmax() for l in results['predictions'].T])
        results.update(compute_metrics(results['final_predictions'], results['labels']))

        # if output_file:
        #     logger.info("***** Save prediction ******")
        #     with open(output_file, 'w') as fout:
        #         pad_token_id = tokenizer.pad_token_id
        #         sentences = sentences.astype(int).tolist()
        #         sentences = [[w for w in s if w != pad_token_id] for s in sentences]
        #         sentences = [tokenizer.convert_ids_to_tokens(s) for s in sentences]
        #         # fout.write('Prediction\tLabel\tSentences\n')
        #         for p, l, s in zip(list(preds), list(out_label_ids), sentences):
        #             s = ' '.join(s)
        #             if label_list:
        #                 p = label_list[p]
        #                 l = label_list[l]
        #             if output_only_prediction:
        #                 fout.write(str(p) + '\n')
        #             else:
        #                 fout.write('{}\t{}\t{}\n'.format(p, l, s))
        logger.info("***** Eval results {} {} *****".format(prefix, language))
        logger.info(f"f1 = {results['f1']}")
        # for key in sorted(results.keys()):
        #     logger.info("  %s = %s", key, str(results[key]))

    return results


def load_and_cache_examples(args, task, preprocessor, split='train', language='en', lang2id=None, evaluate=False,
                            labelled:bool=True, priming_idx:int=-1):
    # Make sure only the first process in distributed training process the
    # dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = PROCESSORS[task]()
    output_mode = "classification"
    # Load data features from cache or dataset file
    lc = '_lc' if args.do_lower_case else ''
    if os.path.exists(args.data_dir):
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}_{}{}".format(
                split,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
                str(language),
                lc,
            ),
        )
    else:
        cached_features_file = os.path.join(
            'download',
            args.data_dir,
            "cached_{}_{}_{}_{}_{}{}".format(
                split,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
                str(language),
                lc,
            ),
        )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if split == 'train':
            examples = processor.get_train_examples(args.data_dir, args.model_name_or_path, language, args.num_sample)
        elif split == 'translate-train':
            examples = processor.get_translate_train_examples(args.data_dir, language)
        elif split == 'translate-test':
            examples = processor.get_translate_test_examples(args.data_dir, language)
        elif split == 'dev':
            examples = processor.get_dev_examples(args.data_dir, args.model_name_or_path, language)
        elif split == 'pseudo_test':
            examples = processor.get_pseudo_test_examples(args.data_dir, language)
        else:
            examples = processor.get_test_examples(args.data_dir, args.model_name_or_path, language)

        if args.retrieval:
            examples = add_priming_data(examples, args.data_dir, args.task_name, args.retriever_name,
                                            args.retrieval_lang, args.num_sim_sents, args.save_sim_sents,
                                            args.retrieval_method, args.seed, args.num_priming, args.random_retrieval)

        features = []
        for (ex_index, example) in tqdm(enumerate(examples), desc='Creating input features from input examples'):
            input_features = preprocessor.get_input_features(example, labelled=labelled, priming_idx=priming_idx,
                                                             priming=args.retrieval, num_priming=args.num_priming,
                                                             stragey=args.aug_strategy)
            features.append(input_features)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the
    # dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

        # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_mlm_labels = torch.tensor([f.mlm_labels for f in features], dtype=torch.long)
    all_idx = torch.tensor([int(f.idx.split('-')[-1]) for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` for {}.".format(args.task_name))

    if args.model_type == 'xlm':
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_langs)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_mlm_labels,
                                all_idx)
    return dataset

def mlm_train_step(batch, preprocessor, model, label_list):
    """
    Perform an MLM training step.
    :return: loss
    """

    inputs = generate_default_inputs(batch)
    mlm_labels, labels = batch['mlm_labels'], batch['labels']

    outputs = model(**inputs)
    prediction_scores = preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
    loss = torch.nn.CrossEntropyLoss()(prediction_scores.view(-1, len(label_list)), labels.view(-1))
    return loss

def mlm_eval_step(batch, preprocessor, model):
    """Perform an MLM evaluation step."""
    inputs = generate_default_inputs(batch)
    outputs = model(**inputs)

    return preprocessor.pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])

def generate_default_inputs(batch, model_type='bert'):
    """Generate the default inputs required by almost every language model."""
    inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
    if model_type in ['bert', 'xlnet']:
        inputs['token_type_ids'] = batch['token_type_ids']
    return inputs


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--train_language", default="en", type=str, help="Train language if is different of the evaluation language."
    )
    parser.add_argument(
        "--predict_languages", type=str, default="en", help="prediction languages separated by ','."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--task_name",
        default="xnli",
        type=str,
        required=True,
        help="The task name",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction.")
    parser.add_argument("--do_predict_dev", action="store_true", help="Whether to run prediction.")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="initial checkpoint for predicting the dev set")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--train_split", type=str, default="train", help="split of training set")
    parser.add_argument("--dev_split", type=str, default='dev', help='split of dev set')
    parser.add_argument("--test_split", type=str, default="test", help="split of test set")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--log_file", default="train", type=str, help="log file")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--eval_test_set",
        action="store_true",
        help="Whether to evaluate test set durinng training",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--early_stopping", action="store_true", 
        help="whether to take early stopping strategy, if dev performance doesn't improve within 3 continuous epochs, then break the training" )
    parser.add_argument("--tolerance", type=str, default=3, help="the maximum number of continuous decreasing epochs")
    parser.add_argument(
        "--save_only_best_checkpoint", action="store_true", help="save only the best checkpoint"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument('--pattern_id', type=int, default='1', help='Prompt template')

    # arguments for few-shot finetuning
    parser.add_argument("--num_sample", type=int, default=-1, help='define the number of shots for few-shot finetuning')

    # arguments w.r.t. retrieval-augmented method
    parser.add_argument('--retrieval', action='store_true', help='use retrieval augmented method')
    parser.add_argument('--num_priming', type=int, default=1, help='how many retrieved samples to use')
    parser.add_argument('--aug_strategy', type=str, default='bow', choices=['bow', 'conc', 'bow_avg'],
                        help="""decide which augmentation strategy to use
                        bow: bag of words, use majority voting
                        conc: concatenation of all retrieved samples
                        bow_avg: averaging the logits of each single retrieval augmented input""")
    parser.add_argument('--retriever_name', type=str, default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                        help='the retriever used for cross-lingual retrieval')
    parser.add_argument('--retrieval_lang', type=str, default='en', help='source language that is used for retrieval')
    parser.add_argument('--num_sim_sents', type=int, default=100, help='the number of retrieved similar sentences')
    parser.add_argument('--save_sim_sents', action='store_true', help='whether to save the similar sentences')
    parser.add_argument('--retrieval_method', type=str, default='sentence_transformer',
                        help='which retrieval method to use')
    parser.add_argument('--random_retrieval', action='store_true', help='whether to use random retrieval')

    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    elif not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logging.basicConfig(
        handlers=[logging.FileHandler(os.path.join(args.output_dir, args.log_file)), logging.StreamHandler()],
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info("Input args: %r" % args)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare dataset
    if args.task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # set save steps to make the save strategy "epoch"
    if args.num_sample != -1:
        args.save_steps = (args.num_sample * num_labels) / (args.n_gpu * args.per_gpu_train_batch_size * args.gradient_accumulation_steps)

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        #args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    logger.info("config = {}".format(config))

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    lang2id = config.lang2id if args.model_type == "xlm" else None
    logger.info("lang2id = {}".format(lang2id))

    # Make sure only the first process in distributed training loads model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)

    preprocessor = MLMPreprocessor(tokenizer, label_list, args.max_seq_length, args.task_name, args.pattern_id)

    # Training
    if args.do_train:
        if args.init_checkpoint:
            logger.info("loading from folder {}".format(args.init_checkpoint))
            model = model_class.from_pretrained(
                args.init_checkpoint,
                config=config,
                cache_dir=args.init_checkpoint,
            )
        else:
            logger.info("loading from existing model {}".format(args.model_name_or_path))
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
        model.to(args.device)

        if args.train_language=='en':
            train_dataset = load_and_cache_examples(args, args.task_name, preprocessor, split=args.train_split,
                                                language=args.train_language, lang2id=lang2id, evaluate=False)
        else:
            train_dataset = load_and_cache_examples(args, args.task_name, preprocessor, split=args.dev_split,
                                                language=args.train_language, lang2id=lang2id, evaluate=False)
        # if trained on full data set, use 'epoch' save strategy
        if args.num_sample == -1:
            args.save_steps = len(train_dataset) // (args.n_gpu * args.per_gpu_train_batch_size * args.gradient_accumulation_steps) + 1

        global_step, tr_loss, best_score, best_checkpoint = train(args, train_dataset, model, tokenizer, preprocessor,
                                                                  label_list, lang2id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info(" best checkpoint = {}, best score = {}".format(best_checkpoint, best_score))

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.init_checkpoint:
        best_checkpoint = args.init_checkpoint
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
    elif args.do_train:
        best_checkpoint = args.output_dir
    else:
        best_checkpoint = None
    best_score = 0
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, preprocessor, split='dev', language=args.train_language, lang2id=lang2id,
                              prefix=prefix)
            if result['f1'] > best_score:
                best_checkpoint = checkpoint
                best_score = result['f1']
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, 'eval_results')
        with open(output_eval_file, 'w') as writer:
            for key, value in results.items():
                writer.write('{} = {}\n'.format(key, value))
            writer.write("Best checkpoint is {}, best f1 is {}".format(best_checkpoint, best_score))
            logger.info("Best checkpoint is {}, best f1 is {}".format(best_checkpoint, best_score))

    # Prediction
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            best_checkpoint if best_checkpoint else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint if best_checkpoint else args.model_name_or_path)
        model.to(args.device)
        output_predict_file = os.path.join(args.output_dir, args.test_split + '_results.txt')
        total = total_correct = 0.0
        with open(output_predict_file, 'a') as writer:
            writer.write('======= Predict using the model from {} for {}:\n'.format(best_checkpoint, args.test_split))
            for language in args.predict_languages.split(','):
                output_file = os.path.join(args.output_dir, 'test-{}.tsv'.format(language))
                result = evaluate(args, model, preprocessor, split=args.test_split, language=language, lang2id=lang2id,
                                  prefix='best_checkpoint' if args.init_checkpoint else args.model_name_or_path)
                writer.write('{}={}\n'.format(language, result['f1']))
                logger.info('{}={}'.format(language, result['f1']))
                total += result['num']
                total_correct += result['correct']
            writer.write('total={}\n'.format(total_correct / total))

    if args.do_predict_dev:
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path if args.model_name_or_path else best_checkpoint, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.init_checkpoint if args.init_checkpoint else args.model_name_or_path)
        model.to(args.device)
        output_predict_file = os.path.join(args.output_dir, 'dev_results')
        total = total_correct = 0.0
        with open(output_predict_file, 'w') as writer:
            writer.write('======= Predict using the model from {}:\n'.format(args.init_checkpoint))
            for language in args.predict_languages.split(','):
                output_file = os.path.join(args.output_dir, 'dev-{}.tsv'.format(language))
                result = evaluate(args, model, preprocessor, split='dev', language=language, lang2id=lang2id,
                                  prefix='best_checkpoint')
                writer.write('{}={}\n'.format(language, result['f1']))
                total += result['num']
                total_correct += result['correct']
            writer.write('total={}\n'.format(total_correct / total))

    return result


if __name__ == "__main__":
    main()

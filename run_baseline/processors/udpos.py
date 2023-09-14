# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" XNLI utils (dataset loading and evaluation) """

import logging
import os
import random

from transformers import DataProcessor
from .utils import InputExample

logger = logging.getLogger(__name__)


class UdposProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_examples(self, data_dir, model, language='en', split='train', num_sample=-1):
        """See base class."""
        examples = []
        for lg in language.split(','):
            # data_file = os.path.join(data_dir, lg, "{}.{}".format(split, model))
            data_file = os.path.join(data_dir, "{}-{}.tsv".format(split, lg))
            with open(data_file, encoding="utf-8") as f:
                words=[]
                labels=[]

                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        if word:
                            examples.append((words,labels,lg))
                            words=[]
                            labels=[]
                    else:
                        splits = line.split("\t")
                        word = splits[0]
                        words.append(splits[0])
                        if len(splits) > 1:
                            labels.append(splits[-1].replace("\n", ""))
                        else:
                            # Examples could have no label for mode = "test"
                            labels.append("O")
                if words:
                    examples.append((words,labels,lg))
        new_example=[]
        index=0
        for example in examples:
            for (i, ex) in enumerate(example[0]):
                guid = "%s-%s-%s" % (split, lg, index)
                index += 1
                text_a = ' '.join('%s' % item for item in example[0])
                text_b = ex
                label = example[1][i]
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                new_example.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=example[2]))

        """
        if num_sample != -1:
            # examples = random.sample(examples, num_sample)
            random.shuffle(examples)
            l0, l1, l2 = [], [], []
            labels = list(set([e.label for e in examples]))
            for example in examples:
                if example.label==labels[0] and len(l0)<num_sample:
                    l0.append(example)
                elif example.label==labels[1] and len(l1)<num_sample:
                    l1.append(example)
                elif example.label==labels[2] and len(l2)<num_sample:
                    l2.append(example)
                elif len(l0)==num_sample and len(l1)==num_sample and len(l2)==num_sample:
                    break
            examples = l0+l1+l2
        """

        return new_example

    def get_train_examples(self, data_dir, model, language='en', num_sample=-1):
        return self.get_examples(data_dir, model, language, split='train', num_sample=num_sample)

    def get_dev_examples(self, data_dir, model, language='en', num_sample=-1):
        return self.get_examples(data_dir, model, language, split='dev', num_sample=num_sample)

    def get_test_examples(self, data_dir, model, language='en', num_sample=-1):
        return self.get_examples(data_dir, model, language, split='test', num_sample=num_sample)

    def get_translate_train_examples(self, data_dir, language='en', num_sample=-1):
        """See base class."""
        examples = []
        for lg in language.split(','):
            file_path = os.path.join(data_dir, "XNLI-Translated/en-{}-translated.tsv".format(lg))
            logger.info("reading file from " + file_path)
            lines = self._read_tsv(file_path)
            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % ("translate-train", lg, i)
                text_a = line[0]
                text_b = line[1]
                label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg))
        if num_sample != -1:
            examples = random.sample(examples, num_sample)
        return examples

    def get_translate_test_examples(self, data_dir, language='en', num_sample=-1):
        lg = language
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-Translated/test-{}-en-translated.tsv".format(lg)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("translate-test", language, i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples

    def get_pseudo_test_examples(self, data_dir, language='en', num_sample=-1):
        lines = self._read_tsv(
            os.path.join(data_dir, "XNLI-Translated/pseudo-test-set/en-{}-pseudo-translated.csv".format(language)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("pseudo-test", language, i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples

    def get_labels(self):
        """See base class."""
        return ["ADJ",
                "ADP",
                "ADV",
                "AUX",
                "CCONJ",
                "DET",
                "INTJ",
                "NOUN",
                "NUM",
                "PART",
                "PRON",
                "PROPN",
                "PUNCT",
                "SCONJ",
                "SYM",
                "VERB",
                "X"]


udpos_processors = {
    "udpos": UdposProcessor,
}

udpos_output_modes = {
    "udpos": "classification",
}

udpos_tasks_num_labels = {
    "udpos": 17,
}

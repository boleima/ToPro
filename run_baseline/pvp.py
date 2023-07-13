"""
This file contains the pattern (prompt template) verbalizer pairs for different tasks.
"""
import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict

from typing import Tuple, Union, List
import torch

from processors.utils import InputExample, get_verbalization_ids

# used for designing the prompt template for data example
FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]

class PVP(ABC):
    """
    This class contains functions to apply patterns and verbalizers as required by prompt learning.
    Each task requires its own custom implementation (processor) of pvp.
    """

    def __init__(self, tokenizer, label_list, max_seq_length, pattern_id: int = 1, verbalizer_file: str = None,
                 seed: int = 42):
        """
        Create a new PVP.

        :param tokenizer: the tokenizer for the underlying language model
        :param label_list: the label list for the task
        :param max_seq_length: maximum sequence length of the model
        :param pattern_id: the pattern id to use
        :param verbalizer_file: an optional file that contains the verbalizer to be used
        :param seed: a seed to be used for generating random numbers if necessary
        """
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.max_seq_length=max_seq_length
        self.pattern_id = pattern_id
        self.rng = random.Random(seed)  # random number generator

        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(verbalizer_file, self.pattern_id)

        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    def _build_mlm_logits_to_cls_logits_tensor(self):
        m2c_tensor = torch.ones([len(self.label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(self.label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.tokenizer, force_single_token=True)
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id

        return m2c_tensor

    @property
    def mask(self) -> str:
        """Return the underlying LM's special mask token."""
        return self.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask token id."""
        return self.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of the verbalizers across all labels."""
        return max(len(self.verbalize(label)) for label in self.label_list)

    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False,
               max_length = None) -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern verbalizer pair

        :param example: an input example to encode
        :param priming: wheather to use this example for priming
        :param labeled: if "priming=True", wheather the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        if not priming:
            assert not labeled, "'labeled' can only be set to true if 'priming' is also set to true."

        tokenizer = self.tokenizer  # type: PreTrainedTokenizer
        parts_a, parts_b = self.get_parts(example)

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts_b if x]

        if max_length:
            self.truncate(parts_a, parts_b, max_length=max_length)
        else:
            self.truncate(parts_a, parts_b, max_length=self.max_seq_length)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part]

        if priming:
            input_ids = tokens_a
            if tokens_b:
                input_ids += tokens_b
            if labeled:
                assert self.mask_id in input_ids, 'sequence of input_ids must contain a mask token'
                mask_idx = input_ids.index(self.mask_id)
                # assert len(self.verbalize(example.label)) == 1, 'priming only supports one verbalization per label'
                verbalizer = self.verbalize(example.label)[0]

                verbalizer_id = get_verbalization_ids(verbalizer, self.tokenizer, force_single_token=True)
                input_ids[mask_idx] = verbalizer_id
            return input_ids, []

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        # input_ids.append(102)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        return input_ids, token_type_ids


    @staticmethod
    def shortenable(s: str) -> Tuple[str, bool]:
        """
        Return an instance of this string that is marked as shortenable
        :param s: the given string to be marked
        :return: a tuple
        """
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark."""
        if isinstance(s, tuple):
            return PVP.remove_final_punc(s[0]), s[1]
        return s.rsplit(string.punctuation)

    # TODO: data type of the first element in the tuple: List[int] or str???
    def truncate(self, parts_a: List[Tuple[List[int], bool]], parts_b: List[Tuple[List[int], bool]], max_length: int):
        """
        Truncate two sequences of text to a predefined total maximum of length.
        :param parts_a: the first text
        :param parts_b: the second text
        :param max_length: predefined total maximum length
        :return: truncated parts_a and parts_b
        """

        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        # total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_a))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor
        m2c = m2c.to(logits.device)

        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels * max_fillers
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()


        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences text_a and text_b, containing exactly one
        mask token for a single task. If a task requires only a single sequence of text, then the second sequence
        should be an empty list.

        :param example: the input example to be processed
        :return: Two sequences of texts. All text segments can optionally be marked as being shortenable.
        """
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label

        :param label: the label
        :return: the list of all verbalizations to the label
        """
        pass

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_ids = None

        with open(path, 'r') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_ids = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_ids][label] = realizations

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize


class XnliPVP(PVP):
    VERBALIZER_A = {
        'entailment': ['Yes'],
        'neutral': ['Maybe'],
        'contradiction': ['No']
    }

    VERBALIZER_B = {
        'entailment': ['Right'],
        'neutral': ['Maybe'],
        'contradiction': ['Wrong']
    }

### VERBALIZER C NOT USABLE ###
    VERBALIZER_C = {
        'entailment': ['implica'],
        'neutral': ['neutral'],
        'contradiction': ['conflict']
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_a = (' '.join(text_a[0]), text_a[1])
        text_b = self.shortenable(example.text_b)

        # pattern 0: text_a. [MASK] text_b
        if self.pattern_id == 0:
            return [text_a, '.', self.mask, text_b], []
        # pattern 1/2: text_a? [MASK], text_b
        elif self.pattern_id == 1 or self.pattern_id == 2:
            return [text_a, '?'], [self.mask, ',', text_b]
        # pattern 3/4： Premise: text_a. Hypothesis: text_b. [MASK]
        elif self.pattern_id == 3 or self.pattern_id == 4:
            return ['Premise: ', text_a, '.' ], ['Hypothesis: ', text_b, '.', self.mask]
        # pattern 5: Premise: text_a. Hypothesis: text_b. The relationship between premise and hypothesis is [MASK].
        elif self.pattern_id == 5:
            return ['Premise: ', text_a, '.' ], ['Hypothesis: ', text_b, '.', 'The relationship between Premise and Hypothesis is', self.mask]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 1 or self.pattern_id == 0 or self.pattern_id==3:
            return XnliPVP.VERBALIZER_A[label]
        elif self.pattern_id == 5:
            return XnliPVP.VERBALIZER_C[label]
        return XnliPVP.VERBALIZER_B[label]


class PawsxPVP(PVP):
    VERBALIZER_A = {
        '0': ["Wrong"],
        '1': ["Right"]
    }
    VERBALIZER_B = {
        '0': ["No"],
        '1': ["Yes"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        
        # patter 0/2: text_a. [mask] text_b
        if self.pattern_id == 0 or 2:
            return [text_a, '.', self.mask, text_b], []
        # patter 1/3: text_a? [mask], text_b
        elif self.pattern_id == 1 or 3:
            return [text_a, '?'], [self.mask, ',', text_b]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or 1:
            return PawsxPVP.VERBALIZER_A[label]
        return PawsxPVP.VERBALIZER_B[label]

class AmazonPVP(PVP):
    VERBALIZER = {
        1: ["terrible"],
        2: ["bad"],
        3: ["ok"],
        4: ["good"],
        5: ["great"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)
        
        # pattern 0: It was [mask]. TEXT
        if self.pattern_id == 0:
            return ['It was', self.mask, '.', text], []
        # pattern 1: TEXT. All in all, it was [mask].
        elif self.pattern_id == 1:
            return [text, '. All in all, it was', self.mask, '.'], []
        # pattern 2: Just [mask]! TEXT
        elif self.pattern_id == 2:
            return ['Just', self.mask, "!"], [text]
        # pattern 3: TEXT. In summary, the product is [mask].
        elif self.pattern_id == 3:
            return [text], ['In summary, the product is', self.mask, '.']
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return AmazonPVP.VERBALIZER[label]


class UdposPVP(PVP):
    VERBALIZER = {
    "ADJ": ["modification"],
    "ADP": ["position"],
    "ADV": ["verbal"],
    "AUX": ["auxiliar"],
    "CCONJ": ["link"],
    "DET": ["determine"],
    "INTJ": ["mode"],
    "NOUN": ["thing"],
    "NUM": ["number"],
    "PART": ["functional"],
    "PRON": ["reference"],
    "PROPN": ["name"],
    "PUNCT": ["punct"],
    "SCONJ": ["condition"],
    "SYM": ["symbol"],
    "VERB": ["verb"],
    "X": ["other"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        
        # pattern 0: TEXT. The tag is [mask].
        if self.pattern_id == 0:
            return [text_a], ['The pos tag of ', text_b, " is a kind of: ", self.mask, ' .']
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return UdposPVP.VERBALIZER[label]

class PanxPVP(PVP):
    VERBALIZER = {
                "B-LOC":["location"],
                "B-ORG":["organisation"],
                "B-PER":["person"],
                "I-LOC":["place"],
                "I-ORG":["body"],
                "I-PER":["name"],
                "O":["other"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        
        # pattern 0: TEXT. The tag is [mask].
        if self.pattern_id == 0:
            return [text_a], ['The named entity of ', text_b, " is a kind of: ", self.mask, ' .']
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return PanxPVP.VERBALIZER[label]


PVPS = {
    'xnli': XnliPVP,
    'pawsx': PawsxPVP,
    'amazon': AmazonPVP,
    'udpos':UdposPVP,
    'panx':PanxPVP
}

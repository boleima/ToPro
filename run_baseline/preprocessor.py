from processors.utils import InputExample, InputFeatures
from pvp import PVPS

from abc import ABC, abstractmethod

class Preprocessor(ABC):
    """
    A preprocessor that transforms an :class:"InputExample" into a :class:"InputFeatures" object so that it can be
    processed by the model being used.
    """
    def __init__(self, tokenizer, label_list, max_seq_length, task_name, pattern_id: int = 1, verbalizer_file: str = None):
        """
        Create a new preprocessor.

        :param tokenizer: the tokenizer for the language model to use
        :param max_seq_length: the maximum sequence length of the model
        :param task_name: the name of the task
        :param pattern_id: the id of prompt patterns to be used
        :param verbalizer_file: path to a file containing a verbalizer that overrides the default verbalizer (optional)
        """

        self.tokenizer = tokenizer
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.pvp = PVPS[task_name](tokenizer, label_list, max_seq_length, pattern_id, verbalizer_file) # pvp stands for patter verbalizer pair
        self.label_map = {label: i for i, label in enumerate(label_list)}
        # convert real label to label index

    @abstractmethod
    def get_input_features(self, example: InputExample, labelled: bool, priming_idx: int = -1, priming: bool = False,
                           **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""
        pass


class MLMPreprocessor(Preprocessor):
    """
    Preprocess for models pretrained using a masked language modeling objective, e.g., BERT.
    """
    def get_input_features(self, example: InputExample, labelled: bool, priming_idx: int = -1, priming: bool = False,
                           num_priming: int=0, strategy: str = 'bow', **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""

        if priming:
            priming_data = example.meta['priming_data'][:num_priming]  # type of priming_data: List[InputExample]
            if strategy=='conc':
                priming_input_ids = []
                max_length = int(self.max_seq_length / (num_priming + 1))
                for priming_example in priming_data:
                    priming_input_ids += \
                        self.pvp.encode(priming_example, priming=True, labeled=True, max_length=max_length)[0]

            else:
                max_length = int(self.max_seq_length / 2)
                priming_example = priming_data[priming_idx]
                priming_input_ids, _ = self.pvp.encode(priming_example, priming=True, labeled=True,
                                                       max_length=max_length)

            input_ids, token_type_ids = self.pvp.encode(example, max_length=max_length)
            input_ids = priming_input_ids + input_ids

            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(input_ids)
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

        else:
            input_ids, token_type_ids = self.pvp.encode(example)

        assert len(input_ids) == len(token_type_ids), f"length of input ids: {len(input_ids)}, " \
                                                      f"length of tokens: {len(token_type_ids)}."

        attention_mask = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)

        if padding_length < 0:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids.")

        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.max_seq_length
        assert len(attention_mask) == self.max_seq_length
        assert len(token_type_ids) == self.max_seq_length

        label = self.label_map[example.label] if example.label is not None else -100  # convert label to label index

        if labelled:
            mlm_labels = self.pvp.get_mask_positions(input_ids)
        else:
            mlm_labels = [-1] * self.max_seq_length

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels, idx=example.guid)


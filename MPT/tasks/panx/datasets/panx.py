# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
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

# Lint as: python3
"""Introduction to the Universal Dependencies 2.2"""

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{pan-etal-2017-cross,
    title = "Cross-lingual Name Tagging and Linking for 282 Languages",
    author = "Pan, Xiaoman  and
      Zhang, Boliang  and
      May, Jonathan  and
      Nothman, Joel  and
      Knight, Kevin  and
      Ji, Heng",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P17-1178",
    doi = "10.18653/v1/P17-1178",
    pages = "1946--1958",
    abstract = "The ambitious goal of this work is to develop a cross-lingual name tagging and linking framework for 282 languages that exist in Wikipedia. Given a document in any of these languages, our framework is able to identify name mentions, assign a coarse-grained or fine-grained type to each mention, and link it to an English Knowledge Base (KB) if it is linkable. We achieve this goal by performing a series of new KB mining methods: generating {``}silver-standard{''} annotations by transferring annotations from English to other languages through cross-lingual links and KB properties, refining annotations through self-training and topic selection, deriving language-specific morphology features from anchor links, and mining word translation pairs from cross-lingual links. Both name tagging and linking results for 282 languages are promising on Wikipedia data and on-Wikipedia data.",
}

"""

_DESCRIPTION = """\
PANX
"""


_DATA_OPTIONS = ['en', 'af', 'ar', 'az', 'bg', 'bn', 'de', 'el', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'jv', 'ka', 'kk', 'ko', 'lt', 'ml', 'mr', 'ms', 'my', 'nl', 'pa', 'pl', 'pt', 'qu', 'ro', 'ru', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yo', 'zh']


_URL = "../../../data/panx/"
_TRAINING_FILE = ".tsv"
_DEV_FILE = ".tsv"
_TEST_FILE = ".tsv"


class PanxConfig(datasets.BuilderConfig):
    """BuilderConfig for Udpos"""

    def __init__(self, **kwargs):
        """BuilderConfig for Panx.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PanxConfig, self).__init__(**kwargs)


class Panx(datasets.GeneratorBasedBuilder):
    """panx dataset."""

    BUILDER_CONFIGS = [
        PanxConfig(name=config_name, version=datasets.Version("0.0.1"), description="PANX Multilingual NER Dataset"
        )
        for config_name in _DATA_OPTIONS
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-PER",
                                "I-PER",
                                "B-ORG",
                                "I-ORG",
                                "B-LOC",
                                "I-LOC",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://aclanthology.org/P17-1178.pdf",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # panx did not provide test set with gold labels, so we use dev sets as test sets.
        urls_to_download = {
            "train": f"{_URL}/train.tsv",
            "dev": f"{_URL}/valid.tsv",
            "test": f"{_URL}/dev-{self.config.name}{_TEST_FILE}",
        }
        print('test config', self.config.name)
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                        chunk_tags = []
                        pos_tags = []
                else:
                    # Udpos tokens are tap separated
                    splits = line.strip().split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1])
                        
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }

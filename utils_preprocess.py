# coding=utf-8
# Copyright 2020 Google and DeepMind.
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
from __future__ import absolute_import, division, print_function

import argparse
from transformers import BertTokenizer, XLMTokenizer, XLMRobertaTokenizer
import os
from collections import defaultdict
import csv
import random
import os
import shutil
import json


TOKENIZERS = {
  'bert': BertTokenizer,
  'xlm': XLMTokenizer,
  'xlmr': XLMRobertaTokenizer,
}

PANX_LANGUAGES = 'ar he vi id jv ms tl eu ml ta te af nl en de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw yo my zh kk tr et fi hu qu pl uk az lt pa gu ro'.split(' ')
UDPOS_LANGUAGES = 'af ar bg de el en es et eu fa fi fr he hi hu id it ja kk ko mr nl pt ru ta te th tl tr ur vi yo zh lt pl uk wo ro'.split(' ')
MEWSLIX_INPUT_LANGUAGES = 'ar de en es fa ja pl ro ta tr uk'.split(' ')

def panx_tokenize_preprocess(args):
  def _preprocess_one_file(infile, outfile, idxfile, tokenizer, max_len):
    if not os.path.exists(infile):
      print(f'{infile} not exists')
      return 0
    special_tokens_count = 3 if isinstance(tokenizer, XLMRobertaTokenizer) else 2
    max_seq_len = max_len - special_tokens_count
    subword_len_counter = idx = 0
    with open(infile, "rt") as fin, open(outfile, "w") as fout, open(idxfile, "w") as fidx:
      for line in fin:
        line = line.strip()
        if not line:
          fout.write('\n')
          fidx.write('\n')
          idx += 1
          subword_len_counter = 0
          continue

        items = line.split()
        token = items[0].strip()
        if len(items) == 2:
          label = items[1].strip()
        else:
          label = 'O'
        current_subwords_len = len(tokenizer.tokenize(token))

        if (current_subwords_len == 0 or current_subwords_len > max_seq_len) and len(token) != 0:
          token = tokenizer.unk_token
          current_subwords_len = 1

        if (subword_len_counter + current_subwords_len) > max_seq_len:
          fout.write(f"\n{token}\t{label}\n")
          fidx.write(f"\n{idx}\n")
          subword_len_counter = current_subwords_len
        else:
          fout.write(f"{token}\t{label}\n")
          fidx.write(f"{idx}\n")
          subword_len_counter += current_subwords_len
    return 1

  model_type = args.model_type
  tokenizer = TOKENIZERS[model_type].from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
  for lang in args.languages.split(','):
    out_dir = os.path.join(args.output_dir, lang)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    if lang == 'en':
      files = ['dev', 'test', 'train']
    else:
      files = ['dev', 'test']
    for file in files:
      infile = os.path.join(args.data_dir, f'{file}-{lang}.tsv')
      outfile = os.path.join(out_dir, "{}.{}".format(file, args.model_name_or_path))
      idxfile = os.path.join(out_dir, "{}.{}.idx".format(file, args.model_name_or_path))
      if os.path.exists(outfile) and os.path.exists(idxfile):
        print(f'{outfile} and {idxfile} exist')
      else:
        code = _preprocess_one_file(infile, outfile, idxfile, tokenizer, args.max_len)
        if code > 0:
          print(f'finish preprocessing {outfile}')


def panx_preprocess(args):
  def _process_one_file(infile, outfile):
    lines = open(infile, 'r').readlines()
    if lines[-1].strip() == '':
      lines = lines[:-1]
    with open(outfile, 'w') as fout:
      for l in lines:
        items = l.strip().split('\t')
        if len(items) == 2:
          label = items[1].strip()
          idx = items[0].find(':')
          if idx != -1:
            token = items[0][idx+1:].strip()
            #if 'test' in infile:
            #  fout.write(f'{token}\n')
            #else:
            fout.write(f'{token}\t{label}\n')
        else:
          fout.write('\n')
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  for lg in PANX_LANGUAGES:
    for split in ['train', 'test', 'dev']:
      infile = os.path.join(args.data_dir, f'{lg}-{split}')
      outfile = os.path.join(args.output_dir, f'{split}-{lg}.tsv')
      _process_one_file(infile, outfile)

def udpos_tokenize_preprocess(args):
  def _preprocess_one_file(infile, outfile, idxfile, tokenizer, max_len):
    if not os.path.exists(infile):
      print(f'{infile} does not exist')
      return
    subword_len_counter = idx = 0
    special_tokens_count = 3 if isinstance(tokenizer, XLMRobertaTokenizer) else 2
    max_seq_len = max_len - special_tokens_count
    with open(infile, "rt") as fin, open(outfile, "w") as fout, open(idxfile, "w") as fidx:
      for line in fin:
        line = line.strip()
        if len(line) == 0 or line == '':
          fout.write('\n')
          fidx.write('\n')
          idx += 1
          subword_len_counter = 0
          continue

        items = line.split()
        if len(items) == 2:
          label = items[1].strip()
        else:
          label = "X"
        token = items[0].strip()
        current_subwords_len = len(tokenizer.tokenize(token))

        if (current_subwords_len == 0 or current_subwords_len > max_seq_len) and len(token) != 0:
          token = tokenizer.unk_token
          current_subwords_len = 1

        if (subword_len_counter + current_subwords_len) > max_seq_len:
          fout.write(f"\n{token}\t{label}\n")
          fidx.write(f"\n{idx}\n")
          subword_len_counter = current_subwords_len
        else:
          fout.write(f"{token}\t{label}\n")
          fidx.write(f"{idx}\n")
          subword_len_counter += current_subwords_len

  model_type = args.model_type
  tokenizer = TOKENIZERS[model_type].from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
  for lang in args.languages.split(','):
    out_dir = os.path.join(args.output_dir, lang)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    if lang == 'en':
      files = ['dev', 'test', 'train']
    else:
      files = ['dev', 'test']
    for file in files:
      infile = os.path.join(args.data_dir, "{}-{}.tsv".format(file, lang))
      outfile = os.path.join(out_dir, "{}.{}".format(file, args.model_name_or_path))
      idxfile = os.path.join(out_dir, "{}.{}.idx".format(file, args.model_name_or_path))
      if os.path.exists(outfile) and os.path.exists(idxfile):
        print(f'{outfile} and {idxfile} exist')
      else:
        _preprocess_one_file(infile, outfile, idxfile, tokenizer, args.max_len)
        print(f'finish preprocessing {outfile}')

def udpos_preprocess(args):
  def _read_one_file(file):
    data = []
    sent, tag, lines = [], [], []
    for line in open(file, 'r'):
      items = line.strip().split('\t')
      if len(items) != 10:
        empty = all(w == '_' for w in sent)
        num_empty = sum([int(w == '_') for w in sent])
        if num_empty == 0 or num_empty < len(sent) - 1:
          data.append((sent, tag, lines))
        sent, tag, lines = [], [], []
      else:
        sent.append(items[1].strip())
        tag.append(items[3].strip())
        lines.append(line.strip())
        assert len(sent) == int(items[0]), 'line={}, sent={}, tag={}'.format(line, sent, tag)
    return data

  def isfloat(value):
    try:
      float(value)
      return True
    except ValueError:
      return False

  def remove_empty_space(data):
    new_data = {}
    for split in data:
      new_data[split] = []
      for sent, tag, lines in data[split]:
        new_sent = [''.join(w.replace('\u200c', '').split(' ')) for w in sent]
        lines = [line.replace('\u200c', '') for line in lines]
        assert len(" ".join(new_sent).split(' ')) == len(tag)
        new_data[split].append((new_sent, tag, lines))
    return new_data

  def check_file(file):
    for i, l in enumerate(open(file)):
      items = l.strip().split('\t')
      assert len(items[0].split(' ')) == len(items[1].split(' ')), 'idx={}, line={}'.format(i, l)

  def _write_files(data, output_dir, lang, suffix):
    for split in data:
      if len(data[split]) > 0:
        prefix = os.path.join(output_dir, f'{split}-{lang}')
        if suffix == 'mt':
          with open(prefix + '.mt.tsv', 'w') as fout:
            for idx, (sent, tag, _) in enumerate(data[split]):
              newline = '\n' if idx != len(data[split]) - 1 else ''
              if split == 'test':
                fout.write('{}{}'.format(' '.join(sent, newline)))
              else:
                fout.write('{}\t{}{}'.format(' '.join(sent), ' '.join(tag), newline))
          check_file(prefix + '.mt.tsv')
          print('  - finish checking ' + prefix + '.mt.tsv')
        elif suffix == 'tsv':
          with open(prefix + '.tsv', 'w') as fout:
            for sidx, (sent, tag, _) in enumerate(data[split]):
              for widx, (w, t) in enumerate(zip(sent, tag)):
                newline = '' if (sidx == len(data[split]) - 1) and (widx == len(sent) - 1) else '\n'
                #if split == 'test':
                #  fout.write('{}{}'.format(w, newline))
                #else:
                fout.write('{}\t{}{}'.format(w, t, newline))
              fout.write('\n')
        elif suffix == 'conll':
          with open(prefix + '.conll', 'w') as fout:
            for _, _, lines in data[split]:
              for l in lines:
                fout.write(l.strip() + '\n')
              fout.write('\n')
        print(f'finish writing file to {prefix}.{suffix}')

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  for root, dirs, files in os.walk(args.data_dir):
    lg = root.strip().split('/')[-1]
    if root == args.data_dir or lg not in UDPOS_LANGUAGES:
      continue

    data = {k: [] for k in ['train', 'dev', 'test']}
    for f in sorted(files):
      if f.endswith('conll'):
        file = os.path.join(root, f)
        examples = _read_one_file(file)
        if 'train' in f:
          data['train'].extend(examples)
        elif 'dev' in f:
          data['dev'].extend(examples)
        elif 'test' in f:
          data['test'].extend(examples)
        else:
          print('split not found: ', file)
        print(' - finish reading {}, {}'.format(file, [(k, len(v)) for k,v in data.items()]))

    data = remove_empty_space(data)
    for sub in ['tsv']:
      _write_files(data, args.output_dir, lg, sub)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_dir", default=None, type=str, required=True,
                      help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--output_dir", default=None, type=str, required=True,
                      help="The output data dir where any processed files will be written to.")
  parser.add_argument("--task", default="panx", type=str, required=True,
                      help="The task name")
  parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str,
                      help="The pre-trained model")
  parser.add_argument("--model_type", default="bert", type=str,
                      help="model type")
  parser.add_argument("--max_len", default=512, type=int,
                      help="the maximum length of sentences")
  parser.add_argument("--do_lower_case", action='store_true',
                      help="whether to do lower case")
  parser.add_argument("--cache_dir", default=None, type=str,
                      help="cache directory")
  parser.add_argument("--languages", default="en", type=str,
                      help="process language")
  parser.add_argument("--remove_last_token", action='store_true',
                      help="whether to remove the last token")
  parser.add_argument("--remove_test_label", action='store_false',
                      help="whether to remove test set label")
  args = parser.parse_args()

  if args.task == 'panx_tokenize':
    panx_tokenize_preprocess(args)
  if args.task == 'panx':
    panx_preprocess(args)
  if args.task == 'udpos_tokenize':
    udpos_tokenize_preprocess(args)
  if args.task == 'udpos':
    udpos_preprocess(args)

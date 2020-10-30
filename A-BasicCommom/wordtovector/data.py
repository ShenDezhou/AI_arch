"""Data processor for SMP-CAIL2020-Argmine.

Author: Yixu GAO (yxgao19@fudan.edu.cn)

In data file, each line contains 1 sc sentence and 5 bc sentences.
The data processor convert each line into 5 samples,
each sample with 1 sc sentence and 1 bc sentence.

Usage:
1. Tokenizer (used for RNN model):
    from data import Tokenizer
    vocab_file = 'vocab.txt'
    sentence = '我饿了，想吃东西了。'
    tokenizer = Tokenizer(vocab_file)
    tokens = tokenizer.tokenize(sentence)
    # ['我', '饿', '了', '，', '想', '吃', '东西', '了', '。']
    ids = tokenizer.convert_tokens_to_ids(tokens)
2. Data:
    from data import Data
    # For training, load train and valid set
    # For BERT model
    data = Data('model/bert/vocab.txt', model_type='bert')
    datasets = data.load_train_and_valid_files(
        'SMP-CAIL2020-train.csv', 'SMP-CAIL2020-valid.csv')
    train_set, valid_set_train, valid_set_valid = datasets
    # For RNN model
    data = Data('model/rnn/vocab.txt', model_type='rnn')
    datasets = data.load_all_files(
        'SMP-CAIL2020-train.csv', 'SMP-CAIL2020-valid.csv')
    train_set, valid_set_train, valid_set_valid = datasets
    # For testing, load test set
    data = Data('model/bert/vocab.txt', model_type='bert')
    test_set = data.load_file('SMP-CAIL2020-test.csv', train=False)
"""
import itertools
import os
# from itertools import islice
from typing import List, Dict, Tuple
# import jieba
import lawa
# import nltk
import torch
from dataclasses import dataclass

import pandas as pd
from gensim.models.word2vec import LineSentence
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import TensorDataset, Dataset
from transformers import BertTokenizer
from tqdm import tqdm
from difflib import SequenceMatcher
import logging
logger = logging.getLogger(__name__)


class Tokenizer:
    """Tokenizer for Chinese given vocab.txt.

    Attributes:
        dictionary: Dict[str, int], {<word>: <index>}
    """
    def __init__(self, language='en', vocab_file='vocab.txt'):
        """Initialize and build dictionary.

        Args:
            vocab_file: one word each line
        """
        self.language = language
        self.dictionary = {'[PAD]': 0, '[UNK]': 1}
        self._pad_token = '[PAD]'
        self.pad_token_id = 0
        count = 2
        with open(vocab_file, encoding='utf-8') as fin:
            for line in fin:
                word = line.strip()
                self.dictionary[word] = count
                count += 1

    def __len__(self):
        return len(self.dictionary)

    def tokenize(self, sentence: str) -> List[str]:
        """Cut words for a sentence.

        Args:
            sentence: sentence

        Returns:
            words list
        """
        if self.language == 'zh':
            words = lawa.lcut(sentence)
        else:  # 'en'
            words = nltk.word_tokenize(sentence)
        return words

    def convert_tokens_to_ids(
            self, tokens_list: List[str]) -> List[int]:
        """Convert tokens to ids.

        Args:
            tokens_list: word list

        Returns:
            index list
        """
        return [self.dictionary.get(w, 1) for w in tokens_list]


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        alines = LineSentence(file_path, limit=block_size)
        self.examples = [tokenizer.convert_tokens_to_ids(line) for line in alines]
        # with open(file_path, encoding="utf-8") as f:
        #     rawlines = []
        #     for line in islice(f, block_size):
        #         line = line.strip()
        #         if (len(line) > 0 and not line.isspace()):
        #             rawlines.append(line)
        #     # lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        #     alines = [list(tokenizer.tokenize(line)) for line in rawlines]
        #     batch_encoding = [tokenizer.convert_tokens_to_ids(line) for line in alines]
        #     self.examples = batch_encoding

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: Tokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    window_size = 5

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            labels = batch.clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        # if self.tokenizer.mask_token is None:
        #     raise ValueError(
        #         "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        #     )

        context_l = []
        for w in range(self.window_size):
            labels = inputs.clone()

            prev_labels = torch.cat([labels[:,w:], labels[:,0:w]], dim=1)
            next_labels = torch.cat([labels[:,-w:], labels[:,0:-w]], dim=1)
            context_l.extend([prev_labels, next_labels])


        # randoms = inputs.clone()
        # indices_random = torch.bernoulli(torch.full(labels.shape)).bool()
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # randoms[indices_random] = random_words[indices_random]



        # # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        # probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # # special_tokens_mask = [
        # #     self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        # # ]
        # # probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        # if self.tokenizer._pad_token is not None:
        #     padding_mask = labels.eq(self.tokenizer.pad_token_id)
        #     probability_matrix.masked_fill_(padding_mask, value=0.0)
        # masked_indices = torch.bernoulli(probability_matrix).bool()
        # labels[~masked_indices] = -100  # We only compute loss on masked tokens
        #
        # # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        # inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        #
        # # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        inputs = torch.cat([inputs] * 2 * self.window_size, dim=0)
        labels = torch.cat(context_l, dim=0)
        inputs = inputs.flatten()
        labels = labels.flatten()
        return inputs, labels




class Data:
    """Data processor for BERT and RNN model for SMP-CAIL2020-Argmine.

    Attributes:
        model_type: 'bert' or 'rnn'
        max_seq_len: int, default: 512
        tokenizer:  BertTokenizer for bert
                    Tokenizer for rnn
    """
    def __init__(self,
                 vocab_file='',
                 max_seq_len: int = 512,
                 model_type: str = 'bert', config=None):
        """Initialize data processor for SMP-CAIL2020-Argmine.

        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
            model_type: 'bert' or 'rnn'
                If model_type == 'bert', use BertTokenizer as tokenizer
                Otherwise, use Tokenizer as tokenizer
        """
        self.model_type = model_type

        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)#BertTokenizer(vocab_file)
        else:  # rnn
            self.tokenizer = Tokenizer(vocab_file=vocab_file)
        self.max_seq_len = max_seq_len
        self.window_size = config.context_window
        self.max_corpus_line = config.max_corpus_line

    def load_file(self,
                  file_path='SMP-CAIL2020-train.csv',
                  train=True) -> TensorDataset:
        """Load SMP-CAIL2020-Argmine train file and construct TensorDataset.

        Args:
            file_path: train file with last column as label
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label

        Returns:
            BERT model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
            RNN model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (s1_ids, s2_ids, s1_length, s2_length, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (s1_ids, s2_ids, s1_length, s2_length)
        """
        sc_list, bc_list, label_list = self._load_file(file_path, train)
        if self.model_type == 'bert':
            dataset = self._convert_sentence_pair_to_bert_dataset(
                sc_list, bc_list, label_list)
        else:  # rnn
            dataset = self._convert_sentence_pair_to_rnn_dataset(
                sc_list, bc_list, label_list)
        return dataset



    def load_train_and_valid_files(self, train_file):
        """Load all files for SMP-CAIL2020-Argmine.

        Args:
            train_file, valid_file: files for SMP-CAIL2020-Argmine

        Returns:
            train_set, valid_set_train, valid_set_valid
            all are torch.utils.data.TensorDataset
        """
        print('Loading train records for train...')
        train_set = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path=train_file,
            block_size=self.max_corpus_line,
        )
        collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )
        # train_set = self.load_file(train_file, True)
        print(len(train_set), 'training records loaded.')
        # print('Loading train records for valid...')
        # valid_set_train = self.load_file(train_file, False)
        # print(len(valid_set_train), 'train records loaded.')
        # print('Loading valid records...')
        # valid_set_valid = self.load_file(valid_file, False)
        # print(len(valid_set_valid), 'valid records loaded.')
        return train_set, collate_fn

    def dynamic_fit_bert_size(self, sc_col, bc_col, max_seq_len):
        if len(sc_col) > max_seq_len // 2 + 1:
            sc_col = sc_col[:max_seq_len//2 + 1]
        if len(bc_col) > max_seq_len // 2:
            bc_col = bc_col[:max_seq_len//2]
        # if len(sc_col) + len(bc_col) > max_seq_len:
        #     sc_summary, bc_summary = self.summarizer.summarize([sc_col, bc_col])
        #     ratio = (len(sc_summary) + len(bc_summary)) * 1.0 / max_seq_len
        #     sc_dest = int(len(sc_summary) // ratio)
        #     bc_dest = int(len(bc_summary) // ratio)
        #     if len(sc_summary) > sc_dest:
        #         x = (len(sc_summary) - sc_dest//3 * 2)//2
        #         sc_summary = sc_summary[:sc_dest//3] +sc_summary[x:x+sc_dest//3] + sc_summary[-sc_dest//3:]
        #     if len(bc_summary) > bc_dest:
        #         x = (len(bc_summary) - sc_dest // 3 * 2) // 2
        #         bc_summary = bc_summary[:bc_dest//3] + bc_summary[x:x+bc_dest//3] +  bc_summary[-bc_dest//3:]
        #     return sc_summary, bc_summary
        return sc_col, bc_col

    def _load_file(self, filename, train: bool = True):
        """Load SMP-CAIL2020-Argmine train/test file.

        For train file,
        The ratio between positive samples and negative samples is 1:4
        Copy positive 3 times so that positive:negative = 1:1

        Args:
            filename: SMP-CAIL2020-Argmine file
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label

        Returns:
            sc_list, bc_list, label_list with the same length
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: List[int], list of labels
        """
        data_frame = pd.read_csv(filename)
        sc_list, bc_list, label_list = [], [], []
        for row in data_frame.itertuples(index=False):
            candidates = row[3:8]
            answer = int(row[-1]) if train else None


            for i, _ in enumerate(candidates):
                sc_summary, bc_summary = self.dynamic_fit_bert_size(row[2], candidates[i], self.max_seq_len)
                sc_tokens = self.tokenizer.tokenize(sc_summary)
                bc_tokens = self.tokenizer.tokenize(bc_summary)

                if train:
                    answer_line = candidates[answer - 1]
                    if len(answer_line) > self.max_seq_len // 2:
                        answer_line = answer_line[:self.max_seq_len // 2]

                    if i + 1 == answer:
                        # Copy positive sample 4 times
                        for _ in range(len(candidates) - 1):
                            sc_list.append(sc_tokens)
                            bc_list.append(bc_tokens)
                            label_list.append(1)
                    else:
                        sc_list.append(sc_tokens)
                        bc_list.append(bc_tokens)

                        #答案相近则标记1
                        meter = SequenceMatcher(None, answer_line, bc_summary).ratio()
                        if meter >= 0.90:
                            label_list.append(1)
                        else:
                            label_list.append(0)

                else:  # test
                    sc_list.append(sc_tokens)
                    bc_list.append(bc_tokens)
        return sc_list, bc_list, label_list

    def _convert_sentence_pair_to_bert_dataset(
            self, s1_list, s2_list, label_list=None):
        """Convert sentence pairs to dataset for BERT model.

        Args:
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
        """
        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for i, _ in tqdm(enumerate(s1_list), ncols=80):
            CONST_LEN = 3 # CLS SEP SEP == 3
            tokens = ['[CLS]'] + s1_list[i] + ['[SEP]']
            segment_ids = [0] * len(tokens)
            tokens += s2_list[i] + ['[SEP]']
            segment_ids += [1] * (len(s2_list[i]) + 1)
            if len(tokens) > self.max_seq_len + CONST_LEN:
                tokens = tokens[:self.max_seq_len + CONST_LEN]
                assert len(tokens) == self.max_seq_len + CONST_LEN
                segment_ids = segment_ids[:self.max_seq_len + CONST_LEN]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            tokens_len = len(input_ids)
            input_ids += [0] * (self.max_seq_len + CONST_LEN - tokens_len)
            segment_ids += [0] * (self.max_seq_len + CONST_LEN - tokens_len)
            input_mask += [0] * (self.max_seq_len + CONST_LEN - tokens_len)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # test
        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids)

    def _convert_sentence_pair_to_rnn_dataset(
            self, s1_list, s2_list, label_list=None):
        """Convert sentences pairs to dataset for RNN model.

        Args:
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
            torch.utils.data.TensorDataset
                each record: (s1_ids, s2_ids, s1_length, s2_length, label)
            Test:
            torch.utils.data.TensorDataset
                each record: (s1_ids, s2_ids, s1_length, s2_length, label)
        """
        all_s1_ids, all_s2_ids = [], []
        all_s1_lengths, all_s2_lengths = [], []
        for i in tqdm(range(len(s1_list)), ncols=80):
            tokens_s1, tokens_s2 = s1_list[i], s2_list[i]
            all_s1_lengths.append(min(len(tokens_s1), self.max_seq_len))
            all_s2_lengths.append(min(len(tokens_s2), self.max_seq_len))
            if len(tokens_s1) > self.max_seq_len:
                tokens_s1 = tokens_s1[:self.max_seq_len]
            if len(tokens_s2) > self.max_seq_len:
                tokens_s2 = tokens_s2[:self.max_seq_len]
            s1_ids = self.tokenizer.convert_tokens_to_ids(tokens_s1)
            s2_ids = self.tokenizer.convert_tokens_to_ids(tokens_s2)
            if len(s1_ids) < self.max_seq_len:
                s1_ids += [0] * (self.max_seq_len - len(s1_ids))
            if len(s2_ids) < self.max_seq_len:
                s2_ids += [0] * (self.max_seq_len - len(s2_ids))
            all_s1_ids.append(s1_ids)
            all_s2_ids.append(s2_ids)
        all_s1_ids = torch.tensor(all_s1_ids, dtype=torch.long)
        all_s2_ids = torch.tensor(all_s2_ids, dtype=torch.long)
        all_s1_lengths = torch.tensor(all_s1_lengths, dtype=torch.long)
        all_s2_lengths = torch.tensor(all_s2_lengths, dtype=torch.long)
        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(
                all_s1_ids, all_s2_ids, all_s1_lengths, all_s2_lengths,
                all_label_ids)
        # test
        return TensorDataset(
            all_s1_ids, all_s2_ids, all_s1_lengths, all_s2_lengths)


def test_data():
    """Test for data module."""
    # For BERT model
    data = Data('model/bert/vocab.txt', model_type='bert')
    _, _, _ = data.load_train_and_valid_files(
        'SMP-CAIL2020-train.csv',
        'SMP-CAIL2020-test1.csv')
    # For RNN model
    data = Data('model/rnn/vocab.txt', model_type='rnn')
    _, _, _ = data.load_train_and_valid_files(
        'SMP-CAIL2020-train.csv',
        'SMP-CAIL2020-test1.csv')


if __name__ == '__main__':
    test_data()

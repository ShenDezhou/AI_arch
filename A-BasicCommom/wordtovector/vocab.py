"""Construct English/Chinese vocab for csv files.

Author: Yixu GAO (yxgao19@fudan.edu.cn)

Use nltk to cut English sentences.
Use jieba to cut Chinese sentences.

Usage:
    from vocab import Vocab
    vocab = Vocab('zh')
    vocab.load_file_to_dict('SMP-CAIL2020-train.csv', list(range(2, 8)))
    vocab.load_file_to_dict('SMP-CAIL2020-test1.csv', list(range(2, 8)))
    vocab.load_file_to_dict('SMP-CAIL2020-test2.csv', list(range(2, 8)))
    vocab.write2file('vocab.txt', False)

In SMP-CAIL2020-Argmine task, simply run this file
    python vocab.py
or
    from vocab import build_vocab
    build_vocab()
"""
from typing import List, Dict

import lawa
import nltk
# nltk.download('punkt')
import pandas as pd


class Vocab:
    """Build vocab for files and write to a file.

    Attributes:
        language: 'zh' for Chinese and 'en' for English
        word_dict: word frequency dict, {<word>: frequency}
            e.g. {'我': 30}
    """

    def __init__(self, language='zh', word_dict: Dict[str, int] = None):
        """Initialize with language type and word_dict.

        Args:
            language: 'zh' or 'en'
        """
        self.language = language
        self.word_dict = word_dict if word_dict else {}

    def load_file_to_dict(
            self, filename: str, cols: List[int] = None) -> Dict[str, int]:
        """Load columns of a csv file to word_dict.

        Args:
            filename: a csv file with ',' as separator
            cols: column indexes to be added to vocab

        Returns:
            word_dict: {<word>: frequency}
        """
        data_frame = pd.read_csv(filename)
        if not cols:
            cols = range(data_frame.shape[1])
        for row in data_frame.itertuples(index=False):
            for i in cols:
                sentence = row[i]
                if self.language == 'zh':
                    words = lawa.lcut(sentence)
                else:  # 'en'
                    words = nltk.word_tokenize(sentence)
                for word in words:
                    self.word_dict[word] = self.word_dict.get(word, 0) + 1
        return self.word_dict

    def write2file(self,
                   filename: str = 'vocab.txt', fre: bool = False, max_vocab_size = None) -> None:
        """Write word_dict to file without file head.
        Each row contains one word with/without its frequency.

        Args:
            filename: usually a txt file
            fre: if True, write frequency for each word
        """
        with open(filename, 'w', encoding='utf-8') as file_out:
            pop_k = [k for k, v in sorted(self.word_dict.items(), key=lambda item: item[1], reverse=True)]
            count = 0
            for pk in pop_k:
                file_out.write(pk)
                if fre:
                    file_out.write(' ' + str(self.word_dict[pk]))
                file_out.write('\n')
                count += 1
                if max_vocab_size and count >= max_vocab_size:
                    break


def build_vocab(file_in, file_out, max_vocab_size= 50000):
    """Build vocab.txt for SMP-CAIL2020-Argmine."""
    vocab = Vocab('en')
    vocab.load_file_to_dict(file_in, None)
    vocab.write2file(file_out, False, max_vocab_size)


if __name__ == '__main__':
    build_vocab('data/train.csv', 'vocab.txt')

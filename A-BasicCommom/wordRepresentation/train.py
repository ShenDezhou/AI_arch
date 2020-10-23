import argparse
import itertools

from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import joblib

from evaluate import eval

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config_file', default='config/bert_config.json',
    help='model config file')

#10.22 experiments showed that 1024 is better than 256
parser.add_argument(
    '-d', '--dimension', default=256,
    help='model vector dimension')

parser.add_argument(
    '-l', '--en', default='en',
    help='corpus is English')

parser.add_argument(
    '-m', '--min_count', default=10,
    help='model vocabulary minimal')

parser.add_argument(
    '-i', '--hierarchical_softmax', default=1,
    help='Hierarchical Softmax')

parser.add_argument(
    '-n', '--negative_sampling', default=5,
    help='Negative Sampling')

parser.add_argument(
    '-w', '--worker', default=8,
    help='model number of workers')

parser.add_argument(
    '-e', '--epoch', default=1,
    help='model number of epochs')

parser.add_argument(
    '-s', '--skip_gram', default=1,
    help='Skip Gram or Continuous Bag of Words')

parser.add_argument(
    '-u', '--analogy_file', default=None,
    help='model validation file')

parser.add_argument(
    '-v', '--similarity_file', default=None,
    help='model validation file')


parser.add_argument(
    '-y', '--window_size', default=5,
    help='model context window size')

parser.add_argument(
   '-z', '--max_vocab_size', default=40000,
    help='model max vocab_size')
args = parser.parse_args()


class LawaIterable(object):
    def __init__(self, filename):
        self.source = filename
        self.limit = None

    def __iter__(self):
        with open(self.source, 'r', encoding='utf-8') as fin:
            for line in itertools.islice(fin, self.limit):
                yield list(lawa.cut(line))



if args.en == "en":
    X = LineSentence(args.config_file)  # 类似迭代器
else:
    import lawa
    X = LawaIterable(args.config_file)

for i in range(1,5):
    model = Word2Vec(X, size=int(args.dimension) * i, min_count=int(args.min_count), window=int(args.window_size), sg=int(args.skip_gram), hs=int(args.hierarchical_softmax) * int(args.negative_sampling), negative=int(args.negative_sampling), workers=int(args.worker), iter=int(args.epoch),
                     max_vocab_size=int(args.max_vocab_size))
    joblib.dump(model, "word2vec.model", compress=3)
    model.wv.save_word2vec_format(fname="word2vec.txt")
    print('vocab:', len(model.wv.vocab.keys()), model.wv.vocab.keys())
    vocab = list(model.wv.vocab.keys())
    print('center word',vocab[-1], model.wv.most_similar(positive=[vocab[-1]], topn=10))

    if args.analogy_file:
        eval(args.analogy_file.split(":"), args.similarity_file.split(":"))


print('FIN')
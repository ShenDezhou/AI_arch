import argparse
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import joblib
import lawa
from evaluate import eval
lawa.load_userdict('dic/Chinese.dic')

lawa.load_userdict('dic/english.dic')

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config_file', default='config/bert_config.json',
    help='model config file')

parser.add_argument(
    '-d', '--dimension', default=256,
    help='model vector dimension')

parser.add_argument(
    '-l', '--en', default=False,
    help='corpus is English')

parser.add_argument(
    '-m', '--min_count', default=1,
    help='model vocabulary minimal')

parser.add_argument(
    '-w', '--worker', default=16,
    help='model number of workers')

parser.add_argument(
    '-e', '--epoch', default=1,
    help='model number of epochs')

parser.add_argument(
    '-u', '--analogy_file', default=None,
    help='model validation file')

parser.add_argument(
    '-v', '--similarity_file', default=None,
    help='model validation file')

args = parser.parse_args()


class LawaIterable(object):
    def __init__(self, lines):
        self.lines = lines

    def __iter__(self):
        for line in self.lines:
            line = line[0]
            yield [" ".join(lawa.lcut(line))]

f = open(args.config_file, 'r', encoding='utf-8')
if args.en == "en":
    X = LineSentence(f)  # 类似迭代器
else:
    X = LineSentence(f)  # 类似迭代器
    X = LawaIterable(X)

for i in range(1,5):
    model = Word2Vec(X, size=args.dimension * i, min_count=int(args.min_count), sg=1, hs=1, workers=int(args.worker), iter=int(args.epoch))
    joblib.dump(model, "word2vec.model", compress=3)
    model.wv.save_word2vec_format(fname="wv.txt")
    print('vocab:', len(model.wv.vocab.keys()), model.wv.vocab.keys())
    vocab = list(model.wv.vocab.keys())
    print('center word',vocab[-1], model.wv.most_similar(positive=[vocab[-1]], topn=10))

    if args.analogy_file:
        eval(args.analogy_file, args.similarity_file)

f.close()
print('FIN')
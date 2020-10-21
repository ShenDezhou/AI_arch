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
    '-min', '--min_count', default=50,
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

X = LineSentence(args.config_file) #类似迭代器

class LawaIterable(object):
    def __init__(self, lines):
        self.lines = lines

    def __iter__(self):
        for line in self.lines:
            line = lawa.lcut(line[0])
            yield " ".join(line)

model = Word2Vec(LawaIterable(X), size=args.dimension, min_count=args.min_count, sg=1, hs=1, workers=args.worker, iter=args.epoch)
# model.build_vocab(X, update=True)
# model.train(X)
print(model.wv.keys())
print(model.wv.most_similar(positive=['be'], topn=10))
# acc=model.accuracy('q.txt')
# print(acc)
joblib.dump(model, "word2vec.model", compress=3)
model.wv.save_word2vec_format(fname="wv.txt")

if args.analogy_file:
    eval(args.analogy_file, args.similarity_file)

print('FIN')
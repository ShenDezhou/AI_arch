import numpy
import pandas
from gensim.models import KeyedVectors
import lawa
from tqdm import tqdm

model = KeyedVectors.load_word2vec_format('model/word2vec.txt', binary=False)
df = pandas.read_csv("data/summary.csv", sep=',', names=['gid','summary','sort'])
adf = df[df.summary!='']
total = len(adf)
dim = model.wv.vector_size
vocab = set(model.wv.vocab.keys())
with open("model/summary-512.txt", 'w', encoding='utf-8') as file:
  file.write(str(total) + " " + str(dim)+'\n')

  for row in tqdm(df.itertuples(index=False)):
    gid = str(row[0])
    summary = str(row[1])
    words = lawa.lcut(summary)
    vecs = numpy.array([model[w] for w in words if w in vocab])
    if len(vecs):
      docvec = numpy.mean(vecs, axis=0)
    else:
      docvec = model['。']
    file.write(gid + " " + numpy.array2string(docvec, separator=' ', max_line_width=10 ** 10, precision=8,
                                                floatmode='fixed', suppress_small=True).lstrip('[').rstrip(']') + '\n')
    #print('.')


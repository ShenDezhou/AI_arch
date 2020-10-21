import joblib
import gensim
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data as tud
# import torch.optim as optim
# from torch.nn.parameter import Parameter

from collections import Counter
import numpy as np
import random
import math
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

def eval(analogy_file, similarity_file):
    model = joblib.load("word2vec.model")
    analog = model.wv.most_similar(positive=['张','三'], negative=['李'], topn=1)
    print(analog)
    similarity = model.wv.similarity('张','李')
    print(similarity)
    #model.wv.save_word2vec_format(fname="wv.txt")

    idx_to_word = [word for word in model.wv.vocab.keys()]
    word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

    # 评估模型
    def evaluate(filename, embedding_weights):
        if filename.endswith('.csv'):
            data = pd.read_csv(filename, sep=',', header=None)
        else:
            if 'analogy' in filename:
                acclist = model.accuracy(filename)
                correct = len(acclist[-1]['correct'])
                total = correct + len(acclist[-1]['incorrect'])
                if total:
                    return correct * 1.0 / total
                return 0
            else:
                data = pd.read_csv(filename, sep='\t', header=None)

        human_similarity = []
        model_similarity = []
        for row in data.itertuples(index=False):
            word1, word2 = row[0], row[1]
            # OOV 不计算精度
            if word1 not in word_to_idx or word2 not in word_to_idx:
                continue
            else:
                # word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
                word1_embed, word2_embed = embedding_weights[[word1]], embedding_weights[[word2]]
                model_similarity.append(float(cosine_similarity(word1_embed, word2_embed)))
                human_similarity.append(float(row[2]))
        spears = scipy.stats.spearmanr(human_similarity, model_similarity)
        return spears.correlation


    evals = evaluate(analogy_file, embedding_weights = model.wv)
    print(evals)
    evals = evaluate(similarity_file, embedding_weights = model.wv)
    print(evals)
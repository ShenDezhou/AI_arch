import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from gensim.models import KeyedVectors

def eval(analogy_file, similarity_file):
    # model = joblib.load("word2vec.model")
    model = KeyedVectors.load_word2vec_format('word2vec.txt', binary=False)
    vocab = list(model.wv.vocab.keys())
    analog = model.wv.most_similar(positive=vocab[0:2], negative=vocab[-1:1], topn=1)
    print('a+b-c',vocab[0:2],vocab[-1],analog)
    similarity = model.wv.similarity(*vocab[0:2])
    print(vocab[0:2],'sim score:', similarity)
    #model.wv.save_word2vec_format(fname="wv.txt")

    idx_to_word = [word for word in model.wv.vocab.keys()]
    word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

    # 评估模型
    def evaluate(filename, embedding_weights):
        miss = 0
        if filename.endswith('.csv'):
            data = pd.read_csv(filename, sep=',', header=None)
            miss += 1
        else:
            if 'analogy' in filename:
                acclist = model.accuracy(filename)
                correct = len(acclist[-1]['correct'])
                total = correct + len(acclist[-1]['incorrect']) + miss
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
                model_similarity.append(0.5)
                human_similarity.append(float(row[2]))
                continue
            else:
                # word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
                word1_embed, word2_embed = embedding_weights[[word1]], embedding_weights[[word2]]
                model_similarity.append(float(cosine_similarity(word1_embed, word2_embed)))
                human_similarity.append(float(row[2]))

        spears = spearmanr(human_similarity, model_similarity)
        return spears.correlation

    total = []
    if isinstance(analogy_file, list):
        for ana in analogy_file:
            evals = evaluate(ana, embedding_weights=model.wv)
            total.append(evals)
            print("analogy score:", evals)
    else:
        evals = evaluate(analogy_file, embedding_weights = model.wv)
        total.append(evals)
        print("analogy score:", evals)

    if isinstance(similarity_file, list):
        for sim in similarity_file:
            evals = evaluate(sim, embedding_weights = model.wv)
            print("similarity score:", evals)
            total.append(evals)
    else:
        evals = evaluate(similarity_file, embedding_weights=model.wv)
        print("similarity score:", evals)
        total.append(evals)

    print("average weighted score:", sum(total))
    return total
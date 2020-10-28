import json
import os
from types import SimpleNamespace

import numpy
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from vocab import build_vocab
from utils import get_path
from model import Word2Vec
from data import Data, LineByLineTextDataset, DataCollatorForLanguageModeling
from evaluate import eval

# def random_batch():
#     random_inputs = []
#     random_labels = []
#     random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)
#
#     for i in random_index:
#         random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # target
#         random_labels.append(skip_grams[i][1])  # context word
#
#     return random_inputs, random_labels

MODEL_MAP = {
    'skipgram': Word2Vec
}

if __name__ == '__main__':
    # batch_size = 2 # mini-batch size
    # embedding_size = 2 # embedding size
    #
    # sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
    #              "dog cat animal", "cat monkey animal", "monkey dog animal"]

    # word_sequence = " ".join(sentences).split()
    # word_list = " ".join(sentences).split()
    # word_list = list(set(word_list))
    # word_dict = {w: i for i, w in enumerate(word_list)}
    # voc_size = len(word_list)
    config_file = 'config/rnn_config.json'

    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    get_path(os.path.join(config.model_path, config.experiment_name))
    get_path(config.log_path)
    # build_vocab(config.train_file_path, os.path.join(config.model_path,'vocab.txt'))

    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                model_type=config.model_type, config=config)
    train_dataset, collate_fn = data.load_train_and_valid_files(
        train_file=config.train_file_path)
    sampler_train = RandomSampler(train_dataset)
    data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler_train,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Make skip gram of one size window
    # skip_grams = []
    # for i in range(1, len(word_sequence) - 1):
    #     target = word_dict[word_sequence[i]]
    #     context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
    #     for w in context:
    #         skip_grams.append([target, w])
    if torch.cuda.is_available():
        device = torch.device('cuda')
        onehot = torch.FloatTensor(numpy.eye(config.vocab_size)).cuda()
    else:
        device = torch.device('cpu')
        onehot = torch.FloatTensor(numpy.eye(config.vocab_size))

    model = MODEL_MAP[config.model_type](config)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    a_score=0
    # Training
    for epoch in range(config.num_epoch):
        tqdm_obj = tqdm(data_loader, ncols=80)

        for step, batch in enumerate(tqdm_obj):
            input_batch, target_batch = batch['input_ids'], batch['labels']
            # input_batch = torch.LongTensor(input_batch)
            # target_batch = torch.LongTensor(target_batch)
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            input_batch = onehot[input_batch]

            optimizer.zero_grad()
            output = model(input_batch)

            # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)

            if a_score:
                tqdm_obj.set_description('anlogy:{:.6f},sim:{:.6f},loss: {:.6f}'.format(a_score, s_score, loss.item()))
            else:
                tqdm_obj.set_description('loss: {:.6f}'.format(loss.item()))

            loss.backward()
            optimizer.step()

            if (step + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
                W, WT = model.parameters()
                weights = W.T.detach().cpu().numpy()
                dic = data.tokenizer.dictionary
                vocab = [key for (key, value) in sorted(dic.items(), key=lambda x: x[1])]
                vocab = numpy.reshape(numpy.array(vocab), (-1, 1))
                w2v = numpy.concatenate((vocab, weights), axis=1)
                pandas.DataFrame(w2v).to_csv("word2vec.txt", sep=' ', header=None, index=False)
                with open("word2vec.txt", 'r+', encoding='utf-8') as file:
                    readcontent = file.read()  # store the read value of exe.txt into
                    file.seek(0, 0)  # Takes the cursor to top line
                    file.write(
                        str(len(vocab)) + " " + str(weights.shape[1]) + "\n")  # convert int to str since write() deals
                    file.write(readcontent)
                # torch.save(model, os.path.join(config.model_path, config.experiment_name, 'model.bin'))

                a_score, s_score = eval(config.analogy_valid_file_path, config.similarity_valid_file_path)
                tqdm_obj.set_description('a{:.6f},s{:.6f},loss: {:.6f}'.format(a_score, s_score, loss.item()))


        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            W, WT = model.parameters()
            weights = W.T.detach().cpu().numpy()
            dic = data.tokenizer.dictionary
            vocab = [key for (key, value) in sorted(dic.items(), key=lambda x: x[1])]
            vocab = numpy.reshape(numpy.array(vocab), (-1, 1))
            w2v = numpy.concatenate((vocab, weights), axis=1)
            pandas.DataFrame(w2v).to_csv("word2vec.txt", sep=' ', header=None, index=False)
            with open("word2vec.txt", 'r+', encoding='utf-8') as file:
                readcontent = file.read()  # store the read value of exe.txt into
                file.seek(0, 0)  # Takes the cursor to top line
                file.write(
                    str(len(vocab)) + " " + str(weights.shape[1]) + "\n")  # convert int to str since write() deals
                file.write(readcontent)
            torch.save(model, os.path.join(config.model_path, config.experiment_name, 'model.bin'))

            a_score, s_score = eval(config.analogy_valid_file_path, config.similarity_valid_file_path)
            tqdm_obj.set_description('a{:.6f},s{:.6f},loss: {:.6f}'.format(a_score, s_score, loss.item()))

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
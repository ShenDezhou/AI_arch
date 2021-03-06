import argparse
import json
import os
from itertools import islice
from types import SimpleNamespace

import numpy
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from vocab import build_vocab
from utils import get_path
from model import Word2Vec,Word2VecHierarchical, Word2VecHierarchicalV2
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
    'skipgram': Word2Vec,
    'hskipgram': Word2VecHierarchicalV2
}

def main(config_file = 'config/hrnn_config.json'):
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

    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    get_path(os.path.join(config.model_path, config.experiment_name))
    get_path(config.log_path)
    # build_vocab(config.train_file_path, os.path.join(config.model_path, 'vocab.txt'), int(config.vocab_size) - 2)

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
    # pos = range(config.vocab_size)
    # i = torch.LongTensor([pos, pos])
    # elements = [1.0] * config.vocab_size
    # v = torch.LongTensor(elements)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # onehot = torch.sparse.FloatTensor(i,v,torch.Size([config.vocab_size, config.vocab_size])).cuda()
    else:
        device = torch.device('cpu')
        # onehot = torch.sparse.FloatTensor(i, v, torch.Size([config.vocab_size, config.vocab_size]))

    model = MODEL_MAP[config.model_type](config)
    # load model states.
    if config.trained_weight:
        model = torch.load(config.trained_weight)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.997)

    a_score = 0
    # Training
    for epoch in range(config.num_epoch):
        tqdm_obj = tqdm(data_loader, ncols=80)

        for step, batch in enumerate(tqdm_obj):
            input_batch, target_batch = batch['input_ids'], batch['labels']
            # input_batch = torch.LongTensor(input_batch)
            # target_batch = torch.LongTensor(target_batch)
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            ind_iter = range(input_batch.shape[0])
            index = 0
            while index < input_batch.shape[0]:
                # use sparse matrix
                # batch_input = None
                # for i_part in islice(ind_iter, index, index + int(config.max_stem_size)):
                #     i_part_input = onehot[input_batch[i_part]].to_dense().unsqueeze(dim=0).float()
                #     if batch_input is not None:
                #         batch_input = torch.cat([batch_input, i_part_input], dim=0)
                #     else:
                #         batch_input = i_part_input
                # 2:
                batch_range = list(islice(ind_iter, index, index + int(config.max_stem_size)))
                batch_input = torch.zeros((len(batch_range), config.vocab_size), dtype=float).float().cuda()
                for i in range(len(batch_range)):
                    batch_input[i, input_batch[batch_range[i]]] = 1.0

                batch_target_batch = target_batch[index: index + int(config.max_stem_size)]
                index += int(config.max_stem_size)

                optimizer.zero_grad()
                output = model(batch_input)
                # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
                if config.hierarchical_softmax:
                    loss = torch.mean(model.hsoftmax(output, batch_target_batch))
                else:
                    loss = criterion(output, batch_target_batch)

                if a_score:
                    tqdm_obj.set_description(
                        'anlogy:{:.6f},sim:{:.6f},loss: {:.6f}'.format(a_score, s_score, loss.item()))
                else:
                    tqdm_obj.set_description('loss: {:.6f}'.format(loss.item()))

                loss.backward()
                optimizer.step()

            if (step + 1) % 100000 == 0:
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
                tqdm_obj.set_description('anlogy:{:.6f},sim:{:.6f},loss: {:.6f}'.format(a_score, s_score, loss.item()))

            # drop the learning rate gradually
            scheduler.step()

        if (epoch + 1) % 1 == 0 or epoch == int(config.num_epoch) - 1:
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
    #
    # for i, label in enumerate(word_list):
    #     W, WT = model.parameters()
    #     x, y = W[0][i].item(), W[1][i].item()
    #     plt.scatter(x, y)
    #     plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config/hrnn_config.json',
        help='model config file')

    parser.add_argument(
        '--local_rank', default=0,
        help='used for distributed parallel')
    args = parser.parse_args()
    main(args.config_file)


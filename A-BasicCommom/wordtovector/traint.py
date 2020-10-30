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
import math
import time
from typing import Dict
import argparse
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import math
import time
from typing import Dict
import argparse
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import numpy
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu


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


def main(config_file='config/bert_config.json'):
    """Main method for training.

    Args:
        config_file: in config dir
    """
    global datasets
    # 0. Load config and mkdir
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))


    get_path(os.path.join(config.model_path, config.experiment_name))
    get_path(config.log_path)
    if config.model_type in ['rnn', 'lr','cnn']:  # build vocab for rnn
        build_vocab(file_in=config.all_train_file_path,
                    file_out=os.path.join(config.model_path, 'vocab.txt'))
    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                model_type=config.model_type, config=config)


    def load_dataset():
        train_dataset, collate_fn = data.load_train_and_valid_files(
            train_file=config.train_file_path)
        return train_dataset, collate_fn

    if config.serial_load:
        train_set, collate_fn = SERIAL_EXEC.run(load_dataset)
    else:
        train_set, collate_fn = load_dataset()


    if torch.cuda.is_available():
        device = torch.device('cuda')
        sampler_train = RandomSampler(train_set)
    else:
        device = torch.device('cpu')
        sampler_train = RandomSampler(train_set)
    # TPU
    device = xm.xla_device()
    sampler_train = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    data_loader = {
        'train': DataLoader(
        train_set,
        batch_size=config.batch_size,
        sampler=sampler_train,
        collate_fn=collate_fn,
        drop_last=True,
    )}

    # 2. Build model
    # model = MODEL_MAP[config.model_type](config)
    model = WRAPPED_MODEL
    #load model states.
    # if config.trained_weight:
    #     model.load_state_dict(torch.load(config.trained_weight))
    model.to(device)
    if torch.cuda.is_available():
        model = model
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, find_unused_parameters=True)




    # # 3. Train
    # trainer = Trainer(model=model, data_loader=data_loader,
    #                   device=device, config=config)
    # # best_model_state_dict = trainer.train()
    #
    # if config.model_type == 'bert':
    #     no_decay = ['bias', 'gamma', 'beta']
    #     optimizer_parameters = [
    #         {'params': [p for n, p in model.named_parameters()
    #                     if not any(nd in n for nd in no_decay)],
    #          'weight_decay_rate': 0.01},
    #         {'params': [p for n, p in model.named_parameters()
    #                     if any(nd in n for nd in no_decay)],
    #          'weight_decay_rate': 0.0}]
    #     optimizer = AdamW(
    #         optimizer_parameters,
    #         lr=config.lr,
    #         betas=(0.9, 0.999),
    #         weight_decay=1e-8,
    #         correct_bias=False)
    # else:  # rnn
    #     optimizer = Adam(model.parameters(), lr=config.lr)

    # if config.model_type == 'bert':
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=config.num_warmup_steps,
    #         num_training_steps=config.num_training_steps)
    # else:  # rnn
    #     scheduler = get_constant_schedule(optimizer)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.997)

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        a_score, s_score = 0, 0
        for x, batch in enumerate(loader):
            input_batch, target_batch = batch['input_ids'], batch['labels']
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            ind_iter = range(input_batch.shape[0])
            index = 0
            while index < input_batch.shape[0]:
                # 2:
                batch_range = list(islice(ind_iter, index, index + int(config.max_stem_size)))
                batch_input = torch.zeros((len(batch_range), config.vocab_size), dtype=float).float().to(device)
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

                loss.backward()
                optimizer.step()
                # drop the learning rate gradually


                if xm.get_ordinal() == 0:
                    if (x + 1) % 100000 == 0:
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
                        print('[xla:{}]({}) anlogy:{:.6f},sim:{:.6f},Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                                xm.get_ordinal(), x, a_score, s_score,loss.item(), tracker.rate(),
                                tracker.global_rate(), time.asctime()), flush=True)
            tracker.add(FLAGS.batch_size)
            scheduler.step()

            if xm.get_ordinal() == 0:
                if (epoch + 1) % 1 == 0 or epoch == int(config.num_epoch) - 1:
                    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
                    W, WT = model.parameters()
                    weights = W.T.detach().cpu().numpy()
                    dic = data.tokenizer.dictionary
                    vocab = [key for (key, value) in sorted(dic.items(), key=lambda x: x[1])]
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
                    print('[xla:{}]({}) anlogy:{:.6f},sim:{:.6f},Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                        xm.get_ordinal(), x, a_score, s_score,loss.item(), tracker.rate(),
                        tracker.global_rate(), time.asctime()), flush=True)
            return a_score, s_score

    # Train and eval loops
    accuracy = 0.0
    data, pred, target = None, None, None
    for epoch in range(FLAGS.num_epoch):
        para_loader = pl.ParallelLoader(data_loader['train'], [device])
        a_score, s_score = train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))

        if FLAGS.metrics_debug:
            xm.master_print(met.metrics_report())

    return a_score, s_score




def _mp_fn(rank, flags, model,serial):
    global WRAPPED_MODEL, FLAGS, SERIAL_EXEC
    WRAPPED_MODEL = model
    FLAGS = flags
    SERIAL_EXEC = serial

    accuracy_valid = main(args.config_file)
    # Retrieve tensors that are on TPU core 0 and plot.
    # plot_results(data.cpu(), pred.cpu(), target.cpu())
    xm.master_print(('DONE',  accuracy_valid))
    # 4. Save model
    if xm.get_ordinal() == 0:
        WRAPPED_MODEL.to('cpu')
        torch.save(WRAPPED_MODEL.state_dict(), os.path.join(config.model_path, 'model.bin'))
        xm.master_print('saved model.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config/lbert_config.json',
        help='model config file')

    parser.add_argument(
        '--local_rank', default=0,
        help='used for distributed parallel')
    args = parser.parse_args()


    with open(args.config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    WRAPPED_MODEL = MODEL_MAP[config.model_type](config)
    if config.trained_weight:
        WRAPPED_MODEL.load_state_dict(torch.load(config.trained_weight))
    FLAGS = config
    SERIAL_EXEC = xmp.MpSerialExecutor()

    # main(args.config_file)
    xmp.spawn(_mp_fn, args=(FLAGS,WRAPPED_MODEL,SERIAL_EXEC,), nprocs=config.num_cores, start_method='fork')

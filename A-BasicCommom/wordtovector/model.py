import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Model
class Word2Vec(nn.Module):
    def __init__(self, config):
        super(Word2Vec, self).__init__()
        # W and WT is not Transpose relationship
        self.config = config
        self.W = nn.Linear(self.config.vocab_size, self.config.embedding_size, bias=False) # voc_size > embedding_size Weight
        self.WT = nn.Linear(self.config.embedding_size, self.config.vocab_size, bias=False) # embedding_size > voc_size Weight

    def forward(self, X):
        # X : [batch_size, voc_size]
        hidden_layer = self.W(X) # hidden_layer : [batch_size, embedding_size]
        output_layer = self.WT(hidden_layer) # output_layer : [batch_size, voc_size]
        return output_layer


import numpy as np
import torch.nn as nn
import torch
import heapq


class Node:
    def __init__(self, token, freq):
        self.vec = torch.randn(300, requires_grad=True, dtype=torch.float)
        self.token = token
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

    def __gt__(self, other):
        return self.freq > other.freq

    def __eq__(self, other):
        if (other == None):
            return False
        if (not isinstance(other, Node)):
            return False
        return self.freq == other.freq


class HuffmanTree:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        self.root = None

    def make_heap(self, frequency):
        for key in frequency:
            node = Node(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = Node(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if (root == None):
            return
        if (root.token != None):
            self.codes[root.token] = current_code
            self.reverse_mapping[current_code] = root.token
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        self.root = root
        current_code = ""
        self.make_codes_helper(root, current_code)

# Model
class Word2VecHierarchical(nn.Module):
    def __init__(self, config, word_frequency):
        super(Word2Vec, self).__init__()
        # W and WT is not Transpose relationship
        self.config = config
        self.W = nn.Linear(self.config.vocab_size, self.config.embedding_size, bias=False) # voc_size > embedding_size Weight
        # self.WT = nn.Linear(self.config.embedding_size, self.config.vocab_size, bias=False) # embedding_size > voc_size Weight
        self.tree = HuffmanTree()
        self.tree.make_heap(word_frequency)
        self.tree.merge_nodes()
        self.tree.make_codes()


    def cal_loss(self, h, target):
        path_to_word = self.tree.codes[target]
        loss = torch.zeros(1, requires_grad=True, dtype=torch.float)
        root = self.tree.root
        for i in path_to_word:
            if (i == '0'):
                loss = loss + torch.log(torch.sigmoid(torch.dot(root.vec, h)))
                root = root.left
            else:
                loss = loss + torch.log(torch.sigmoid(-1 * torch.dot(root.vec, h)))
                root = root.right
        loss = loss * -1
        return loss

    def forward(self, X):
        # X : [batch_size, voc_size]

        hidden_layer = self.W(X) # hidden_layer : [batch_size, embedding_size]
        # output_layer = self.WT(hidden_layer) # output_layer : [batch_size, voc_size]
        return hidden_layer

#
#
# def word_count(corpus):
#     counter = [0] * len(corpus.dictionary.idx2word)
#     for i in corpus.train:
#         counter[i] += 1
#     for i in corpus.valid:
#         counter[i] += 1
#     for i in corpus.test:
#         counter[i] += 1
#     return np.array(counter).astype(int)
#
# def word_freq_ordered(corpus):
#     # Given a word_freq_rank, we could find the word_idx of that word in the corpus
#     counter = word_count(corpus = corpus)
#     # idx_order: freq from large to small (from left to right)
#     idx_order = np.argsort(-counter)
#     return idx_order.astype(int)
#
# def word_rank_dictionary(corpus):
#     # Given a word_idx, we could find the frequency rank (0-N, the smaller the rank, the higher frequency the word) of that word in the corpus
#     idx_order = word_freq_ordered(corpus = corpus)
#     # Reverse
#     rank_dictionary = np.zeros(len(idx_order))
#     for rank, word_idx in enumerate(idx_order):
#         rank_dictionary[word_idx] = rank
#     return rank_dictionary.astype(int)
#
#
#
# class Rand_Idxed_Corpus(object):
#     # Corpus using word rank as index
#     def __init__(self, corpus, word_rank):
#         self.dictionary = self.convert_dictionary(dictionary = corpus.dictionary, word_rank = word_rank)
#         self.train = self.convert_tokens(tokens = corpus.train, word_rank = word_rank)
#         self.valid = self.convert_tokens(tokens = corpus.valid, word_rank = word_rank)
#         self.test = self.convert_tokens(tokens = corpus.test, word_rank = word_rank)
#
#     def convert_tokens(self, tokens, word_rank):
#         rank_tokens = torch.LongTensor(len(tokens))
#         for i in range(len(tokens)):
#             rank_tokens[i] = int(word_rank[tokens[i]])
#         return rank_tokens
#
#     def convert_dictionary(self, dictionary, word_rank):
#         rank_dictionary = data.Dictionary()
#         rank_dictionary.idx2word = [''] * len(dictionary.idx2word)
#         for idx, word in enumerate(dictionary.idx2word):
#
#             rank = word_rank[idx]
#             rank_dictionary.idx2word[rank] = word
#             if word not in rank_dictionary.word2idx:
#                 rank_dictionary.word2idx[word] = rank
#         return rank_dictionary
#
#
#
# class Word2VecEncoder(nn.Module):
#
#     def __init__(self, ntoken, ninp, dropout):
#         super(Word2VecEncoder, self).__init__()
#         self.drop = nn.Dropout(dropout)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         self.init_weights()
#
#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, input):
#
#         emb = self.encoder(input)
#         emb = self.drop(emb)
#         return emb
#
# class LinearDecoder(nn.Module):
#     def __init__(self, nhid, ntoken):
#         super(LinearDecoder, self).__init__()
#         self.decoder = nn.Linear(nhid, ntoken)
#         self.init_weights()
#
#     def init_weights(self):
#         initrange = 0.1
#         self.decoder.bias.data.fill_(0)
#         self.decoder.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, inputs):
#         decoded = self.decoder(inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2)))
#         return decoded.view(inputs.size(0), inputs.size(1), decoded.size(1))
#
#
# class HierarchicalSoftmax(nn.Module):
#     def __init__(self, ntokens, nhid, ntokens_per_class = None):
#         super(HierarchicalSoftmax, self).__init__()
#
#         # Parameters
#         self.ntokens = ntokens
#         self.nhid = nhid
#
#         if ntokens_per_class is None:
#             ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))
#
#         self.ntokens_per_class = ntokens_per_class
#
#         self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
#         self.ntokens_actual = self.nclasses * self.ntokens_per_class
#
#         self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
#         self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)
#
#         self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class), requires_grad=True)
#         self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)
#
#         self.softmax = nn.Softmax(dim=1)
#
#         self.init_weights()
#
#     def init_weights(self):
#
#         initrange = 0.1
#         self.layer_top_W.data.uniform_(-initrange, initrange)
#         self.layer_top_b.data.fill_(0)
#         self.layer_bottom_W.data.uniform_(-initrange, initrange)
#         self.layer_bottom_b.data.fill_(0)
#
#
#     def forward(self, inputs, labels = None):
#
#         batch_size, d = inputs.size()
#
#         if labels is not None:
#
#             label_position_top = labels / self.ntokens_per_class
#             label_position_bottom = labels % self.ntokens_per_class
#
#             layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
#             layer_top_probs = self.softmax(layer_top_logits)
#
#             layer_bottom_logits = torch.squeeze(torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + self.layer_bottom_b[label_position_top]
#             layer_bottom_probs = self.softmax(layer_bottom_logits)
#
#             target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]
#
#             return target_probs
#
#         else:
#             # Remain to be implemented
#             layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
#             layer_top_probs = self.softmax(layer_top_logits)
#
#             word_probs = layer_top_probs[:,0] * self.softmax(torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0])
#
#             for i in range(1, self.nclasses):
#                 word_probs = torch.cat((word_probs, layer_top_probs[:,i] * self.softmax(torch.matmul(inputs, self.layer_bottom_W[i]) + self.layer_bottom_b[i])), dim=1)
#
#             return word_probs
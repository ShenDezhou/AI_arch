import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import heapq



# Model
from torch.autograd import Variable


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


class Node:
    def __init__(self, token, freq, leaf_size):
        self.vec = torch.randn(leaf_size, requires_grad=True, dtype=torch.float).cuda()
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
    def __init__(self, leaf_size):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        self.root = None
        self.leaf_size = leaf_size

    def make_heap(self, frequency):
        for key in frequency:
            node = Node(key, frequency[key], self.leaf_size)
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = Node(None, node1.freq + node2.freq, self.leaf_size)
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
    def __init__(self, config):
        super(Word2VecHierarchical, self).__init__()
        # W and WT is not Transpose relationship
        self.config = config
        self.W = nn.Linear(self.config.vocab_size, self.config.embedding_size, bias=False) # voc_size > embedding_size Weight
        # self.WT = nn.Linear(self.config.embedding_size, self.config.vocab_size, bias=False) # embedding_size > voc_size Weight
        self.tree = HuffmanTree(self.config.embedding_size)
        # for huffman tree build a frequecy vocabuary.
        word_frequency = dict(zip(range(self.config.vocab_size), range(self.config.vocab_size+1,1, -1)))
        self.tree.make_heap(word_frequency)
        self.tree.merge_nodes()
        self.tree.make_codes()

        self.value_to_path_and_nodes_dict = {}
        for key, value in word_frequency.items():
            codes = self.tree.codes[key]
            self.value_to_path_and_nodes_dict[key] = codes


    def cal_loss(self, hidden_layer, target):
        path_to_word = self.value_to_path_and_nodes_dict[target.item()]
        loss = torch.zeros(1, requires_grad=True, dtype=torch.float).cuda()
        root = self.tree.root
        for i in path_to_word:
            if i == '0':
                loss = loss + torch.log(torch.sigmoid(torch.dot(root.vec, hidden_layer)))
                root = root.left
            else:
                loss = loss + torch.log(torch.sigmoid(-1 * torch.dot(root.vec, hidden_layer)))
                root = root.right
        loss = loss * -1
        return loss

    def cal_losses(self, hidden_layer, target):
        losses = torch.cat([self.cal_loss(hidden_layer[t], target[t]) for t in range(len(target))], dim=0)
        return torch.sum(losses)


    def forward(self, X):
        # X : [batch_size, voc_size]
        hidden_layer = self.W(X) # hidden_layer : [batch_size, embedding_size]
        # output_layer = self.WT(hidden_layer) # output_layer : [batch_size, voc_size]
        return hidden_layer

class HierarchicalSoftmax(nn.Module):
    def __init__(self, ntokens, nhid, ntokens_per_class = None):
        super(HierarchicalSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nhid = nhid

        if ntokens_per_class is None:
            ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))

        self.ntokens_per_class = ntokens_per_class

        self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
        self.ntokens_actual = self.nclasses * self.ntokens_per_class

        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)

        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class), requires_grad=True)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):

        initrange = 0.1
        self.layer_top_W.data.uniform_(-initrange, initrange)
        self.layer_top_b.data.fill_(0)
        self.layer_bottom_W.data.uniform_(-initrange, initrange)
        self.layer_bottom_b.data.fill_(0)


    def forward(self, inputs, labels):

        batch_size, d = inputs.size()
        label_position_top = labels // self.ntokens_per_class
        label_position_bottom = labels % self.ntokens_per_class

        layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
        layer_top_probs = self.softmax(layer_top_logits)

        layer_bottom_logits = torch.squeeze(torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + self.layer_bottom_b[label_position_top]
        layer_bottom_probs = self.softmax(layer_bottom_logits)

        target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]
        return target_probs
        #
        # else:
        #     # Remain to be implemented
        #     layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
        #     layer_top_probs = self.softmax(layer_top_logits)
        #
        #     word_probs = layer_top_probs[:,0] * self.softmax(torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0])
        #
        #     for i in range(1, self.nclasses):
        #         word_probs = torch.cat((word_probs, layer_top_probs[:,i] * self.softmax(torch.matmul(inputs, self.layer_bottom_W[i]) + self.layer_bottom_b[i])), dim=1)
        #
        #     return word_probs

# Model
class Word2VecHierarchicalV2(nn.Module):
    def __init__(self, config):
        super(Word2VecHierarchicalV2, self).__init__()
        # W and WT is not Transpose relationship
        self.W = nn.Linear(config.vocab_size, config.embedding_size, bias=False) # voc_size > embedding_size Weight
        # self.WT = nn.Linear(self.config.embedding_size, self.config.vocab_size, bias=False) # embedding_size > voc_size Weight
        self.hsoftmax = HierarchicalSoftmax(config.vocab_size, config.embedding_size)

    def forward(self, X):
        # X : [batch_size, voc_size]
        hidden_layer = self.W(X) # hidden_layer : [batch_size, embedding_size]
        # output_layer = self.WT(hidden_layer) # output_layer : [batch_size, voc_size]
        return hidden_layer

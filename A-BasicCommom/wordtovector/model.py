import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Model
class Word2Vec(nn.Module):
    def __init__(self, config):
        super(Word2Vec, self).__init__()
        # W and WT is not Traspose relationship
        self.config = config
        self.W = nn.Linear(self.config.vocab_size, self.config.embedding_size, bias=False) # voc_size > embedding_size Weight
        self.WT = nn.Linear(self.config.embedding_size, self.config.vocab_size, bias=False) # embedding_size > voc_size Weight

    def forward(self, X):
        # X : [batch_size, voc_size]
        hidden_layer = self.W(X) # hidden_layer : [batch_size, embedding_size]
        output_layer = self.WT(hidden_layer) # output_layer : [batch_size, voc_size]
        return output_layer
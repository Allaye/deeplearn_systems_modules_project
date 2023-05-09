# Description: classes to build the architecture of the language detector model
# author: Kolade Gideon @Allaye 2023
# github: github.com/Allaye
# created: 2023-03-05
# last modified: 2023-03-28

import torch
import torch.nn as nn
import gensim.models as gsm
from torch.utils.data import DataLoader

from dataloader import yield_tokens, collate_batch


class EmbeddingTrainer:

    def __init__(self, data, **kwargs: dict):
        self.data = data
        self.kwargs = kwargs

    def train_all(self):
        train_en = self.train_en(self.data)
        train_fr = self.train_fr(self.data)
        train_yor = self.train_yor(self.data)
        train_ewe = self.train_ewe(self.data)
        return train_en, train_fr, train_yor, train_ewe

    @staticmethod
    def train_en(data):
        data = [d for d in yield_tokens(data, 'eng')]
        return gsm.Word2Vec(sentences=data, min_count=1)

    @staticmethod
    def train_fr(data):
        data = [d for d in yield_tokens(data, 'fra')]
        return gsm.Word2Vec(sentences=data, min_count=1)

    @staticmethod
    def train_yor(data):
        data = [d for d in yield_tokens(data, 'yor')]
        return gsm.Word2Vec(sentences=data, min_count=1)

    @staticmethod
    def train_ewe(data):
        data = [d for d in yield_tokens(data, 'ewe')]
        return gsm.Word2Vec(sentences=data, min_count=1)


class EmbeddingDetector(nn.Module):

    def __init__(self, vocab_size, hidden_state_size, num_lang, pretrained_embeddings=None):
        super(EmbeddingDetector, self).__init__()
        self.hidden_size = hidden_state_size
        self.emb1 = nn.EmbeddingBag(vocab_size, hidden_state_size, sparse=True)
        self.emb1.weight.data.copy_(pretrained_embeddings)
        self.fc1 = nn.Linear(hidden_state_size, num_lang)
        # self.fc2 = nn.Linear(128, num_lang)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.6
        self.emb1.weight.data.uniform_(-initrange, initrange)
        # self.emb2.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        # self.fc2.weight.data.uniform_(-initrange, initrange)
        # self.fc2.bias.data.zero_()

    def forward(self, text, offsets=None):
        embedded = self.emb1(text, offsets)  # + self.emb2(text, offsets)
        embedded = embedded.view(embedded.size(0), -1)
        # print(embedded.shape)
        embedded = self.fc1(embedded)
        # embedded = self.sigmoid(embedded)
        return embedded


class RNNDetector(nn.Module):
    def __init__(self, vocab_size, hidden_state_size, num_layers, num_lang, pretrained_embeddings=None):
        super(RNNDetector, self).__init__()
        self.hidden_size = hidden_state_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(vocab_size, hidden_state_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_state_size, num_lang)
        self.softmax = nn.Softmax()

    def init_weight(self):
        return torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float32)

    def forward(self, x, offsets=None):
        h0 = self.init_weight()

        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        # out = self.softmax(out)
        return out


class LSTMDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.hidden_size)
        # print('shape 1', h0.shape, c0.shape, x.shape, self.hidden_size)
        #
        # print('shape', h0.shape, c0.shape)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out.reshape(lstm_out.shape[0], -1)

        # attention_out = self.attention(lstm_out)
        fc_out = self.fc(lstm_out)
        return fc_out


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_weights = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        attn_weights = self.attn_weights(lstm_output)
        attn_weights = self.softmax(attn_weights)
        attn_applied = torch.bmm(attn_weights.transpose(1,2), lstm_output)
        return attn_applied

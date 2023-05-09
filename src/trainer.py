# Description: functions to prepare the model and perform training
# author: Kolade Gideon @Allaye 2023
# github: github.com/Allaye
# created: 2023-03-05
# last modified: 2023-03-28


import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from langdector import EmbeddingDetector, RNNDetector, LSTMDetector, EmbeddingTrainer
from dataloader import CustomSentenceDataset, collate_batch1, prepare_dataset, yield_tokens
from utils import hyperparameter_tuning, total_vocab, save_embedding, load_embedding

num_epochs, learning_rate, batch_size, num_workers, device = hyperparameter_tuning()
vocab_size, output = total_vocab()
train, test, df = prepare_dataset()
train_dataset = CustomSentenceDataset(train)
test_dataset = CustomSentenceDataset(test)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch1)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch1)
fre_emb, eng_emb, yor_emb, ewe_emb = load_embedding('../model/embeddings/', ['fr.wv', 'en.wv', 'yor.wv', 'ewe.wv'])
print('vector', ewe_emb.wv.vectors.shape)
pretrained_embeddings = torch.concat([torch.from_numpy(np.asarray(fre_emb.wv.vectors)),
                                      torch.from_numpy(np.asarray(eng_emb.wv.vectors)),
                                      torch.from_numpy(np.asarray(yor_emb.wv.vectors)),
                                      torch.from_numpy(np.asarray(ewe_emb.wv.vectors))], dim=0)
print('pretrained', pretrained_embeddings.shape)
print('vocab', vocab_size, output)
embedding = EmbeddingDetector(pretrained_embeddings.shape[0], 100, output, pretrained_embeddings)


def train(model, dataloader, val_dataloader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    total_acc, total_count = 0, 0
    for epoch in range(num_epochs):
        for idx, data in enumerate(dataloader):

            target, sentence, offset = data
            sentence_fixed = sentence.clone()
            sentence_fixed[sentence >= pretrained_embeddings.shape[0]] = 0
            optimizer.zero_grad()
            sentence_fixed = sentence_fixed.to(torch.float32)
            # print('sentence', sentence_fixed.shape, sentence_fixed.dtype)
            predicted = model(sentence_fixed)
            # print('predicted', torch.argmax(predicted, dim=1), target)
            loss = criterion(predicted, target)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(embedding.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted.argmax(1) == target).sum().item()
            total_count += target.size(0)
            if idx % 100 == 0 and idx > 0:
                print(f'Epoch: {epoch}, Step: {idx}, Accuracy: {total_acc / total_count}')
                total_acc, total_count = 0, 0
        model.eval()
        with torch.no_grad():
            val_acc, val_count = 0, 0
            for data in val_dataloader:
                target, sentence, offset = data
                sentence_fixed = sentence.clone()
                sentence_fixed[sentence >= pretrained_embeddings.shape[0]] = 0
                sentence_fixed = sentence_fixed.to(torch.float32)
                predicted = model(sentence_fixed)
                val_acc += (predicted.argmax(1) == target).sum().item()
                val_count += target.size(0)
            print(f'Epoch: {epoch}, Validation Accuracy: {val_acc / val_count}')

    # torch.save(model.state_dict(), '../model/embedding.pt')


def train_kfold(model, dataset):
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold}')
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch1)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch1)
        train(model, train_loader, val_loader)


def train_embeddings(df):
    et = EmbeddingTrainer(df)
    models = et.train_all()
    save_embedding(models, '../model/embeddings/', ['en.wv.', 'fr.wv', 'yor.wv', 'ewe.wv'])


if __name__ == '__main__':
    kfold = KFold(n_splits=10, shuffle=True)
    emb = EmbeddingDetector(pretrained_embeddings.shape[0], 100, output, pretrained_embeddings)
    rnn = RNNDetector(pretrained_embeddings.shape[0], 100, 2, output, pretrained_embeddings)
    lstm = LSTMDetector(pretrained_embeddings.shape[0], 100, 2, output)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    # for idx, data in enumerate(dataloader):
    #     target, sentence, offset = data
    #     print('target', sentence.shape, offset.shape, target.shape, data)
    #     print('sentense', sentence.type())
    #     print('data', data[0].shape, data[1].shape, data[2].shape)
    #     print('target', target)
    #     sentence = sentence.to(torch.float32)
    #     prediction = rnn(sentence)

    # break
    # print('prediction', prediction)

    train_kfold(lstm, train_dataset)
    # train_embeddings(df)

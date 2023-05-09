# Description: Helper functions for the project
# author: Kolade Gideon @Allaye 2023
# github: github.com/Allaye
# created: 2023-03-05
# last modified: 2023-03-28
import pickle
import torch
from sklearn.model_selection import KFold
from trainer import train
from torch.utils.data import DataLoader, Subset
from dataloader import vocabularies


def language_inference():
    """
        This function is used to test the language model.
        It takes a sentence as input and returns the next word.
    """
    pass


def load_checkpoint(model, checkpoint_path):
    """"
        This function is used to load the model from a checkpoint.
        It takes the model and the path to the checkpoint as input.
        It returns the model.
    """
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    except FileNotFoundError:
        print("====>Checkpoint not found. Starting from scratch.")


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
        This function is used to save the model to a checkpoint.
        It takes the model and the path to the checkpoint as input.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def save_metrics(filename, train_loss, train_acc, val_loss, val_acc):
    """
        This function is used to save the metrics to a csv file.
        It takes the filename, train_loss, train_acc, val_loss, val_acc as input.
    """
    with open(filename, 'w') as f:
        f.write("train_loss,train_acc,val_loss,val_acc\n")
        for i in range(len(train_loss)):
            f.write(f"{train_loss[i]},{train_acc[i]},{val_loss[i]},{val_acc[i]}\n")


def train_kfold(model, dataset, batch_size, fn, kfold=KFold(10, True, 1)):
    """
        This function is used to evaluate the model using kfold cross validation.
    """
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold}')
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=fn)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=fn)
        train(model, train_loader, val_loader)


def total_vocab():
    """
        This function is used to get the total vocabulary of the dataset.
    :return: total vocabulary
    """
    total = 0
    [total := total + len(lang) for lang in vocabularies.values()]
    return total, len(vocabularies)


def hyperparameter_tuning():
    num_epochs = 50
    learning_rate = 0.0001  # karpathy's constant
    batch_size = 20
    num_workers = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return num_epochs, learning_rate, batch_size, num_workers, device


def save_embedding(models, path, filename):
    """
        This function is used to save the embedding.
        It takes the embedding model and file name as input.
        It returns the saved embedding.
    """
    for model, filename in zip(models, filename):
        with open(path + filename, 'wb') as f:
            model.save(f)
    return None


def load_embedding(path, filename):
    """
        This function is used to load the embedding.
        It takes the embedding model and file name as input.
        It returns the loaded embedding.
    """
    weights = []
    for file in filename:
        with open(path + file, 'rb') as f:
            weights.append(pickle.load(f))
    return weights


print(total_vocab())
# weights = load_embedding('../model/embeddings/', ['fr.wv', 'en.wv'])
# print(len(weights[0].wv.key_to_index))
# print(len(weights[1].wv.key_to_index))
# print(len(weights))
# print(weights[1].wv['women'] - weights[0].wv['femmes'])
#
# from gensim.parsing.preprocessing import preprocess_string
# from gensim.parsing.preprocessing import strip_numeric, strip_punctuation, strip_multiple_whitespaces, strip_short, remove_stopwords
# from functools import reduce
#
# f = 'Yehowa na ɖevi si Uriya srɔ̃ dzi na David la'
# s = '4 "Dzidzɔtɔwoe nye ame siwo le nu xam, elabena'
# t = 'Bí àwòṣe ìdẹ́rùbà rẹ bá ní ìjọba tàbí agbófinr 3ni4'
# # print(preprocess_string(f),  strip_numeric(s))
# func = [strip_numeric, strip_punctuation, strip_multiple_whitespaces, strip_short, remove_stopwords]
# print(reduce(lambda x, y: y(x), func, t))

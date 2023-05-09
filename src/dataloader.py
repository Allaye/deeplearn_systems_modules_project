# Description: functions to prepare the dataset for training
# author: Kolade Gideon @Allaye 2023
# github: github.com/Allaye
# created: 2023-03-05
# last modified: 2023-03-17

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from gensim.parsing.preprocessing import strip_numeric, strip_punctuation, strip_multiple_whitespaces, strip_short, remove_stopwords
from functools import reduce
from sklearn.model_selection import train_test_split

from spacy.lang.yo import Yoruba


# data analysis and wrangling
def load_data(uri="../data/sentences1.csv"):
    """Load the dataset from the file and return the dataset"""
    return pd.read_csv('../data/sentences.csv')


def get_diff_lang(dataframe):
    """Get the different languages in the dataset"""
    return dataframe['lan_code'].unique()


def get_diff_lang_count(dataframe):
    """Get the number of occurrence of all different languages in the dataset"""
    return dataframe.lan_code.value_counts()


def return_lang_specific_count(dataframe, count=1000):
    """
    Return the number of occurrence of all different languages in the dataset
    that are greater than the count
    """
    # df[df['lang_code'].isin(df['lang_code'].value_counts()[df['lang_code'].value_counts() > 1000].index)]
    # Compute value counts of lang_code
    lang_counts = dataframe['lan_code'].value_counts()

    # Filter lang_codes with count > 1000
    popular_langs = lang_counts[lang_counts > 1000].index

    # Slice dataframe to include only popular langs
    df_filtered = dataframe[dataframe['lan_code'].isin(popular_langs)]
    return df_filtered


def remove_nan(dataframe):
    """Remove all rows with nan values"""
    return dataframe.dropna()


def return_lang_to_be_used(dataframe, lang=['ewe', 'yor', 'eng', 'fra']):
    """ Return the languages to be used
    In general, you will need enough training data to capture the variation and complexity of each language you are trying to identify.
    For some languages, e4 a small amount of training data may be sufficient, while for others, you may need a larger amount
    of data.

    As a rule of thumb, a minimum of 1000 to 2000 sentences per language is recommended for training a language identification
    model. However, this can vary depending on the language and the quality of the data.

    Args: dataframe: the dataframe to be used
          lang: the languages TO BE RETURNED    : the dataframe with the languages to be used
    """
    return dataframe[dataframe['lan_code'].isin(['fra', 'eng', 'yor', 'ewe'])]


def drop_columns(dataframe, columns=['id']):
    """Drop the columns that are not needed"""
    # check if id column is in the dataframe
    if 'id' in dataframe.columns:
        return dataframe.drop(columns, axis=1, inplace=True)
    return dataframe


def prepare_dataset(dataframe=None):
    """Prepare the dataset for training"""
    # remove all rows with nan values
    if dataframe is None:
        dataframe = load_data()
    dataframe = remove_nan(dataframe)
    # drop the columns that are not needed
    # dataframe = drop_columns(dataframe)
    # return the languages to be used
    dataframe = return_lang_to_be_used(dataframe)
    # convert lan_code to numerical values
    # dataframe['code'] = pd.factorize(dataframe['lan_code'])[0] + 1
    group = dataframe.groupby('lan_code')
    # df_percent = dataframe.sample(frac=0.7, random_state=42)
    french = group.get_group('fra').sample(n=20000, random_state=42)
    english = group.get_group('eng').sample(n=20000, random_state=42)
    yoruba = group.get_group('yor')
    ewe = group.get_group('ewe')
    dataframe = pd.concat([french, english, yoruba, ewe])
    dataframe.lan_code = pd.Categorical(dataframe.lan_code)
    dataframe['code'] = dataframe.lan_code.cat.codes
    # dataframe.to_csv('../data/sentences.csv', index=False)
    # # return eng, fre, yor, ewe languages
    train, test = train_test_split(dataframe, test_size=0.2, random_state=0)
    train.to_csv('../data/train.csv', index=False)
    test.to_csv('../data/test.csv', index=False)
    return train, test, dataframe


#
# dataframe = load_data()
# # # #
#
# #
# print(train.head(5))
# print(test.head(5))
# train_set, test_set = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
# test_set = DataLoader(test_set, batch_size=1, shuffle=True)
# for i in test_set:
#     print(i)
#     break

# def yield_tokens(data_iter, lang: str = 'eng'):
#     """Yield tokens"""
#     # df1 = data_iter[data_iter['lan_code'] == lang]
#
#     for _, text in iter(data_iter):
#         yield text
#
#
# vocab = build_vocab_from_iterator(yield_tokens(train_set), specials=["<unk>"])
# vocab.set_default_index(vocab["<unk>"])

#
# def collate_fn(batch):
#     """Collate a batch of data"""
#     # Sort a data list by text length (descending order).
#     batch.sort(key=lambda x: len(x[0]), reverse=True)
#     # Separate source and target sequences.
#     sentences, lan_code = zip(*batch)
#     # Merge sequences (from tuple of 1D tensor to 2D tensor).
#     sentences = pad_sequence(sentences, padding_value=PAD_IDX)
#     return sentences, torch.tensor(lan_code)
#

# print("language", languages.shape)
# print("desc", languages.describe())
# print("english", languages[1].shape)
# print("yoruba", languages[2].shape)
# print("ewe", languages[3].shape)

tokenizer_func = {
    'eng': get_tokenizer('spacy', language='en_core_web_sm'),
    'fra': get_tokenizer('spacy', language='fr_core_news_sm'),
    'yor': get_tokenizer(None),
    'ewe': get_tokenizer(None)
}
vocabularies = {}
text_pipeline = {}

# Define special symbols and indices
UNK_IDX = 0
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_token = ['<unk>']

# yor_tokenizer = get_tokenizer('spacy', language='yo')
# ewe_tokenizer = get_tokenizer('spacy', language='ee')
import spacy


#
# nlp = Yoruba()  # use directly
# snlp = spacy.blank("yo")  # blank instance
# print('yo', type(snlp('Egbe ni owo')))
# tok = list(tokenizer.get('yor')('Egbe ni owo'))
# print('yor', type(tok))
# print('yor', tok)
# print('yor', tok[0])
# print('yor', tok[1])
# print('yor', tok[2])
#
# print('yoruba', list(tokenizer_func.get('yor')('Egbe ni owo')))
# print('ewe', tokenizer_func.get('ewe')('va mi dzo'))
# print('french', tokenizer_func.get('fra')('je suis un homme'))
# print('english', tokenizer_func.get('eng')('how are you doing'))


class CustomSentenceDataset(Dataset):
    """Custom dataset"""

    def __init__(self, dataframe):
        super(CustomSentenceDataset, self).__init__()
        # self.dataframe = pd.read_csv(path).drop(columns=['lan_code'], axis=1)
        # self.dataframe = pd.read_csv(path)
        self.dataframe = dataframe
        self.feature = self.dataframe['sentence']
        self.lang = self.dataframe['lan_code']
        self.label = self.dataframe['code']

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.feature.iloc[idx], self.label.iloc[idx], self.lang.iloc[idx]
        # return self.dataframe.iloc[idx]


def yield_tokens(data_iter, lang: str = 'eng'):
    """Yield tokens"""
    df_filtered = data_iter[data_iter['lan_code'] == lang]
    for _, data in df_filtered.iterrows():
        tokenizer = tokenizer_func.get(lang)
        func = [strip_numeric, strip_punctuation, strip_multiple_whitespaces, strip_short, remove_stopwords]
        sentence = reduce(lambda x, y: y(x), func, data.sentence.lower())
        yield tokenizer(sentence)


train, test, df = prepare_dataset()
data = CustomSentenceDataset(train)
dataframe = data.dataframe

for lang in ['eng', 'fra', 'yor', 'ewe']:
    vocab = build_vocab_from_iterator(yield_tokens(dataframe, lang), specials=special_token)
    vocab.set_default_index(vocab["<unk>"])
    vocabularies[lang] = vocab
    text_pipeline[lang] = lambda x, lang: vocab(tokenizer_func.get(lang)(x))


def sentence_pipeline(text, lang):
    token = tokenizer_func.get(lang)(text.lower())
    return vocabularies.get(lang)(token)


# ni
# ní
# ní

# print('yoruba', sentence_pipeline('ìfẹsẹ̀wọnsẹ̀ kúuṣẹ́” fifa', 'yor'))
# print(vocabularies.get('yor')(['ìfẹsẹ̀wọnsẹ̀', 'kúuṣẹ́”', 'fifa']))
# # print(text_pipeline.get('yor')('ìfẹsẹ̀wọnsẹ̀ kúuṣẹ́” fifa', 'yor'))
# vocab = build_vocab_from_iterator(yield_tokens(dataframe, 'yor'), specials=special_token)
# vocab.set_default_index(vocab["<unk>"])
# print(vocab(['ìfẹsẹ̀wọnsẹ̀', 'kúuṣẹ́”', 'fifa']))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch, padding=False):
    label_list, text_list, offsets = [], [], [0]
    # if padding:
    max_length = max([len(sentence_pipeline(_text, _lang)) for (_text, _label, _lang) in batch])
    for (_text, _label, _lang) in batch:
        label_list.append(_label)
        # print('text', _text)
        processed_text = torch.tensor(sentence_pipeline(_text, _lang), dtype=torch.int64)

        processed_text = F.pad(processed_text, (0, 68943 - len(processed_text)))
        # print('processed_text', processed_text)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets


def collate_batch1(batch):
    label_list, text_list, offsets = [], [], [0]
    max_length = max([len(sentence_pipeline(_text, _lang)) for (_text, _label, _lang) in batch])
    for (_text, _label, _lang) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(sentence_pipeline(_text, _lang), dtype=torch.int64)
        processed_text = F.pad(processed_text, (0, 68943 - len(processed_text)))
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.stack(text_list)
    return label_list, text_list, offsets
# data = CustomSentenceDataset('../data/sentences.csv')
# print('testingddsfgdfvdg', len(data))
# print('testing', data[0])
# dataloader = DataLoader(data, batch_size=4, shuffle=True, collate_fn=collate_batch)
# #
# print('testing')
# for d in dataloader:
#     print('collate data', d)
#     break
# dataframe = data.dataframe
#
# for lang in ['eng', 'fra', 'yor', 'ewe']:
#     vocab = build_vocab_from_iterator(yield_tokens(dataframe, lang), specials=special_token)
#     vocab.set_default_index(vocab["<unk>"])
#     vocabularies[lang] = vocab

# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
#
# tokenizer = get_tokenizer('basic_english')

# train_dataset, test_dataset  = AG_NEWS()

# vocab = build_vocab_from_iterator(yield_tokens(dataframe, 'yor'), specials=special_token)
# vocab.set_default_index(vocab["<unk>"])


# def yield_token(data_iter):
#     for _, text in data_iter:
#         yield tokenizer(text)


# vocab1 = build_vocab_from_iterator(yield_token(dataframe), specials=["<unk>"])
# vocab1.set_default_index(vocab["<unk>"])

# print('vocab', vocab(['fakọyọ', 'najme']))
# print('vocab1', vocab1(['my', 'najme']))


# batch.sort(key=lambda x: len(x[0]), reverse=True)
# # Separate source and target sequences.
# sentences, lan_code = zip(*batch)
# # Merge sequences (from tuple of 1D tensor to 2D tensor).
# sentences = pad_sequence(sentences, padding_value=PAD_IDX)
# return sentences, torch.tensor(lan_code)


# print('vocan', vocabularies)
# print('eng vocab', vocabularies.get('eng').get_stoi()['said', 'that', 'she', 'was'])

# rint('yor vocab', vocabularies.get('yor')(['said', 'iṣẹ́', 'àwọn', 'ṣiṣé̩', 'wasnbjghfgdfchgj']))
# vocab = build_vocab_from_iterator(yield_tokens(dataframe), specials=special_token)
# # vocab.set_default_index(vocab["<unk>"])
# print('yor vocab', vocabularies.get('yor').get_stoi())
# print('vocab', vocabularies.get('yor').get_stoi().get('wasnbjghfgdfchgj', '<unk>'))
# print('yor vocab', vocabularies.get('yor').forward(['said', 'iṣẹ́', 'àwọn', 'ṣiṣé̩', 'wasnbjghfgdfchgj']))
# #
# print('eng vocab', eng_vocab)
# #
# print(data)
# print(data[0])
# print('data', data.dataframe.head(3))
# print(tokenizer.get('yor'))

# dataset = pd.read_csv('../data/sentences.csv').drop(columns=['lan_code'], axis=1)
# feature = dataset['sentence']
# label = dataset['code']
# print(feature.head())
# print(label.head())

# for token in yield_tokens(data):
#     print(token)
#     break

# lan = dataframe[dataframe['lan_code'] == 'yor'].head(3)
# for _, data in lan.iterrows():
#     print(data['sentence'])
#     print(data['lan_code'])
#     print(data.code)
#
#     break

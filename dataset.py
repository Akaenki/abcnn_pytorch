# -*- coding: utf-8 -*-

import os

import numpy as np
from torch.utils.data import Dataset
from konlpy.tag import Twitter 
from gensim.models import KeyedVectors, Word2Vec

#load pretrained word vector
try:
    model = Word2Vec.load('/usr/bin/ko.min.bin')
except:
    model = Word2Vec.load('ko.min.bin')

twitter = Twitter()
word_vectors = model.wv
unknowns = {}

def trainTestSplit(dataset_path, max_length, val_share=0.003):
    
    queries_path = os.path.join(dataset_path, 'train', 'train_data')
    labels_path = os.path.join(dataset_path, 'train', 'train_label')
        
    with open(queries_path, 'rt', encoding='utf8') as f:
        queries = preprocess(f.readlines(), max_length)

    with open(labels_path) as f:
        labels = np.array([[np.float32(x)] for x in f.readlines()])

    val_offset = int(len(queries)*(1-val_share))
    dataset = (queries, labels)

    return KinQueryDataset(dataset, 0, val_offset), KinQueryDataset(dataset, val_offset, len(queries)-val_offset)

class KinQueryDataset(Dataset):
    def __init__(self, dataset_path, max_length, emb_dim):

        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')

        with open(queries_path, 'rt', encoding='utf8') as f:
            self.queries = preprocess(f.readlines(), max_length, emb_dim)

        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]

def vectorize(sentence, emb_dim):
    sentences = sentenceDivide(sentence)
    
    return (clear(sentences[0], emb_dim), clear(sentences[1], emb_dim))

def clear(sentence, emb_dim):
    vectors = []
    remove = ['><', '-_-', '--', "!", "?", ".", ",", "\"", ";", "\^", "_", "\/", "★", "♥", "ㅠ", "~"]
    special = "내공"
    
    for i in remove:
        if i in sentence:
            sentence = sentence.replace(i, "")
    if "(" in sentence and ")" in sentence:
        left = sentence.index("(")
        right = sentence.index(")")
        if "내공" in sentence[left: right]:
            sentence = sentence[:left] + sentence[right+1:]

    tokens = twitter.pos(sentence, stem=True)
    for token, tag in tokens:
        if tag == 'Verb' or tag == 'Adjective':
            token = token[:-1]
        try:
            vectors.append(word_vectors[token])
        except:
            if token in unknowns:
                vectors.append(unknowns[token])
            else:
                unknowns[token] = np.random.uniform(-0.01, 0.01, emb_dim).astype("float32")
                vectors.append(unknowns[token])
    
    return vectors

def sentenceDivide(sentence):
    sentences = sentence.split('\t')
    s1 = sentences[0]
    s2 = sentences[1].replace('\n', '')

    return (s1, s2)

def preprocess(data, max_length, emb_dim):
    vectorized_data = [[], []]
    
    for datum in data:
        vectors = vectorize(datum, emb_dim)
        vectorized_data[0].append(vectors[0])
        vectorized_data[1].append(vectors[1])
    zero_padding = np.zeros((len(data), 2, max_length, emb_dim), dtype=np.float32)
    
    for i in range(2):
        for idx, seq in enumerate(vectorized_data[i]):
            length = len(seq)
            
            if length >= max_length:
                length = max_length
                zero_padding[idx, i, :length, :] = np.array(seq)[:length]

            else:
                zero_padding[idx, i, :length, :] = np.array(seq)

    return zero_padding
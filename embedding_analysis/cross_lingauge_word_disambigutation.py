# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:30:24 2022

@author: perni
"""

from datetime import datetime
import re
import pandas as pd
import numpy as np
import math
import random

import spacy

import torch
from torch.nn import CosineSimilarity

from transformers import BertTokenizerFast, BertModel

import sys
sys.path.append("../utils/")

from model_utils import *
from load_dict import * 

# pick layer for intermediate comparison
LAYER = 7

# define constants
DATASET_FOLDER = "D:/datasets/"
SAVING_FOLDER = "../data/"


## Load data
word_tensors = torch.tensor(np.load("../data/embeddings/average_embeddings.npy"), dtype=torch.float32)
dict_words = np.load("../data/embeddings/word_to_embedding.npy", allow_pickle=True)

# load english word-embeddings
word_tensors_en = torch.tensor(np.load("../data/embeddings/average_embeddings_en.npy"), dtype=torch.float32)
dict_words_en = np.load("../data/embeddings/word_to_embedding_en.npy", allow_pickle=True)

# load bert-model
model_name = 'NbAiLab/nb-bert-base'
tokenizer, bert_model = load_bert(model_name, is_cuda = True)

## Helper functions
def find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = False, top = 10): 
    """Find the most similar words to a word in context

    Parameters
    ----------
    word_array : array
        word tokenizer input context
    word_idx: int
        index of the relevant word in the word_array
    lang_ad: bool
        define if we are adjusting for language characteristics or not

    Returns
    -------
    array
        top most similar words
    array
        the similarity score to the top words
    """
    word_embedding = torch.tensor(find_word_embedding_idx(word_idx, word_array, tokenizer, bert_model, LAYER = LAYER))
    if (lang_ad): 
        word_embedding =  word_embedding - en_mean_vector + no_mean_vector
    
    return get_most_similar_from_vector(word_embedding, word_tensors[:, LAYER, :], dict_words, is_cuda = False, top = top)


## Language specific data

with open(SAVING_FOLDER+"en2no_test_dictionary.json", "r") as f: 
    en2no_dict = json.load(f)

# create language vector
no_mean_vector = torch.mean(word_tensors[:, LAYER, :], 0)
en_mean_vector = torch.mean(word_tensors_en[:, LAYER, :], 0)


## Test translations

# English sentence with "do"
sent = "Can you please do it ? "
word = "do"
word_array = sent.split()
word_idx = word_array.index(word)
print(sent, word)
print("Closest words: ")
print(find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = True, top = 3))

# Norwegian sentence with "do"
sent = "Jeg må gå på do en tur . "
word = "do"
word_array = sent.split()
word_idx = word_array.index(word)
print(sent, word)
print("Closest words: ")
print(find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = False, top = 3))

# English sentence with "make"
sent = "Should we be focusing on replicating them – or trying to make new, tasty alternatives ? "
word = "make"
word_array = sent.split()
word_idx = word_array.index(word)
print(sent, word)
print("Closest words: ")
print(find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = True, top = 3))

# Norwegian sentence with "do"
sent = "Blant noen dyr setter belønningssystemet i gang, når dyrene finner sammen med en make . "
word = "make"
word_array = sent.split()
word_idx = word_array.index(word)
print(sent, word)
print("Closest words: ")
print(find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = False, top = 3))

# English sentence with "love"
sent = "Do you love me ? "
word = "love"
word_array = sent.split()
word_idx = word_array.index(word)
print(sent, word)
print("Closest words: ")
print(find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = True, top = 3))

# Norwegian sentence with "love"
sent = "Kan du love meg at vi ikke behøver å vente tre timer i kø ? "
word = "love"
word_array = sent.split()
word_idx = word_array.index(word)
print(sent, word)
print("Closest words: ")
print(find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = False, top = 3))
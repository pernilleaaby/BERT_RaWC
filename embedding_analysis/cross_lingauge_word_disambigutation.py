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

import nltk

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
def find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = False, to_language = "no", top = 10): 
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
    if (lang_ad and to_language == "no"): 
        word_embedding =  word_embedding - en_mean_vector + no_mean_vector
    elif(lang_ad): 
        word_embedding =  word_embedding - no_mean_vector + en_mean_vector
        
    if (to_language == "no"): 
        most_similar_matches = get_most_similar_from_vector(word_embedding, word_tensors[:, LAYER, :], dict_words, is_cuda = False, top = top)
    else: 
        most_similar_matches = get_most_similar_from_vector(word_embedding, word_tensors_en[:, LAYER, :], dict_words_en, is_cuda = False, top = top)

    return most_similar_matches


## Language specific data

with open(SAVING_FOLDER+"en2no_test_dictionary.json", "r") as f: 
    en2no_dict = json.load(f)

# create language vector
no_mean_vector = torch.mean(word_tensors[:, LAYER, :], 0)
en_mean_vector = torch.mean(word_tensors_en[:, LAYER, :], 0)


## Test translations
shared_words = ['do', 'be', 'love', 'far', 'to']
en_sents = [("Can you please do it? ", "do"), 
            ("The two offices could be combined to achieve better efficiency and reduce the cost of administration.", "be"), 
            ("Do you love me? ", "love"), 
            ("Mr. Reama , far from really being retired , is engaged in industrial relations counseling.", "far"), 
            ("It was hard to do the job without any supervision.", "to")
            ]
no_sents = [("Jeg må gå på do en tur.", "do"), 
            ("Hans la igjen Batman kostyme hjemme for å gå og be om knask eller knep i noe litt med kosete.", "be"), 
            ("Kan du love meg at vi ikke behøver å vente tre timer i kø?", "love"), 
            ("Det å bli far til en jente, endret synet mitt på en del ting.", "far"), 
            ("Hun fikk to fingre delvis amputert.", "to")
    ]

for en_sent in en_sents: 
    word_array = nltk.word_tokenize(en_sent[0])
    word_idx = word_array.index(en_sent[1])
    print(en_sent[0], en_sent[1])
    print("Closest words Norwegian: ")
    print(find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = True, to_language="no", top = 3)[0])
    print("Closest words English: ")
    print(find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = False, to_language="en", top = 3)[0])
    print()
    
for no_sent in no_sents: 
    word_array = nltk.word_tokenize(no_sent[0])
    word_idx = word_array.index(no_sent[1])
    print(no_sent[0], no_sent[1])
    print("Closest words English: ")
    print(find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = True, to_language="en", top = 3)[0])
    print("Closest words Norwegian: ")
    print(find_knn_words_array(word_array, word_idx, tokenizer, bert_model, lang_ad = False, to_language="no", top = 3)[0])
    print()

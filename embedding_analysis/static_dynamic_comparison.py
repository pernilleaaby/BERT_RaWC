# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:26:32 2022

@author: perni
"""

import re
import pandas as pd
import numpy as np
import math
import random

import torch
from torch.nn import CosineSimilarity

from transformers import BertTokenizerFast, BertModel
from datasets import load_dataset

import sys
sys.path.append("../utils/")

from model_utils import *

# define constants
LAYER = 8

SAVING_FOLDER = "../data/"

# load word-embeddings
word_tensors = torch.tensor(np.load("../data/embeddings/average_embeddings.npy"), dtype=torch.float32)
dict_words = np.load("../data/embeddings/word_to_embedding.npy", allow_pickle=True)

# load bert-model
model_name = 'NbAiLab/nb-bert-base'
tokenizer, bert_model = load_bert(model_name, is_cuda = True)

###############################
# Helper functions
###############################

def find_word_embedding(word, word_array, tokenizer, bert_model): 
    token_mapping, token_embeddings = tokens_to_embeddings(word_array, tokenizer, bert_model)
    word_idx = word_array.index(word)
    
    # obtain embeddings for word
    token_ids = np.where(token_mapping == word_idx)[0]
    word_embedding = np.mean(token_embeddings[token_ids], axis=0)[LAYER]
    
    return word_embedding

def find_word_embedding_idx(word_idx, word_array, tokenizer, bert_model): 
    token_mapping, token_embeddings = tokens_to_embeddings(word_array, tokenizer, bert_model)
    
    # obtain embeddings for word
    token_ids = np.where(token_mapping == word_idx)[0]
    word_embedding = np.mean(token_embeddings[token_ids], axis=0)[LAYER]
    
    return word_embedding

# define cosine function 
cos = CosineSimilarity(dim=1)

def get_cosine_sims(word_tensor): 
    cos_sims = cos(word_tensor, word_tensors[:, LAYER, :])
    # make to numpy
    cos_sims = cos_sims.detach().numpy()
    
    return cos_sims

def get_most_similar_from_vector(word_vector, top = 10): 
    word_vector = word_vector.unsqueeze(0)
    # fetch cosine similarities
    cos_sims = get_cosine_sims(word_vector)
    sorted_indecies = np.argsort(cos_sims)
    
    return dict_words[sorted_indecies[-top:]][::-1], cos_sims[sorted_indecies][-top:][::-1]

def find_embedding(word): 
    word_index = list(dict_words).index(word)
    if (type(word_index) == int): 
        return word_tensors[word_index]
    else: 
        return 0

def get_most_similar(word, top = 10): 
    word_index = list(dict_words).index(word)
    word_tensor = torch.tensor(word_tensors[word_index, LAYER].clone().detach().numpy(), dtype=torch.float32).unsqueeze(0)
    
    # fetch cosine similarities
    cos_sims = get_cosine_sims(word_tensor)
    sorted_indecies = np.argsort(cos_sims)
    
    return dict_words[sorted_indecies[-top-1:-1]][::-1], cos_sims[sorted_indecies][-top-1:-1][::-1]

def find_knn_words(sent, word, tokenizer, bert_model, top = 10): 
    word_array = [token.text for token in nlp(sent)]
    word_embedding = torch.tensor(find_word_embedding(word, word_array, tokenizer, bert_model))
    
    return get_most_similar_from_vector(word_embedding, top = top)

def find_knn_words_array(word_array, word_idx, tokenizer, bert_model, top = 10): 
    word_embedding = torch.tensor(find_word_embedding_idx(word_idx, word_array, tokenizer, bert_model))
    
    return get_most_similar_from_vector(word_embedding, top = top)

def confuse_tokens(tokenized_chunk): 
    # extract indecies of relevant candidate words
    relevant_words = []
    for i, word in enumerate(tokenized_chunk): 
        if (word.lower() in confusion_sets): 
            relevant_words.append(i)
         
    # return false if no confusion candidates
    if (len(relevant_words)==0): 
        return False

    # pick random word
    word_candidate_idx = random.choice(relevant_words)
    word_candidate = tokenized_chunk[word_candidate_idx].lower()
    # pick random substitution candidate
    sub_candidate = random.choice(confusion_sets[word_candidate])

    confusion_tuple = (word_candidate, sub_candidate, word_candidate_idx)

    # switch words and make into new sentence
    tokenized_chunk[word_candidate_idx] = sub_candidate
    
    return tokenized_chunk, confusion_tuple

def find_relevant_word_ids(tokenized_chunk): 
    # extract indecies of relevant candidate words
    relevant_words = []
    for i, word in enumerate(tokenized_chunk): 
        if (word.lower() in confusion_sets): 
            relevant_words.append(i)
            
    return relevant_words

def find_knn_words_array_multi(word_array, relevant_ids, tokenizer, bert_model, top = 10):
    most_similar_words = []
    for word_idx in relevant_ids: 
        word_embedding = torch.tensor(find_word_embedding_idx(word_idx, word_array, tokenizer, bert_model))
        most_similar_word = get_most_similar_from_vector(word_embedding, top = top)[0][0]
        most_similar_words.append(most_similar_word)
    
    return most_similar_words


#########################
# Running program
#########################


## Load confusion set
with open(SAVING_FOLDER+"words_with_confusions.json", "r") as f: 
    confusion_sets = json.load(f)

## Load in Treebank
norne = load_dataset('NbAiLab/norne', 'bokmaal')

### Words in wrong context
m_itself = 0
m_original = 0
m_other = 0
confusion_results = []
for train_idx in range(len(norne['train'])): 
    train_row = norne['train'][train_idx]
    confusion = confuse_tokens(train_row['tokens'])
    if (not confusion): # did not find any words to substitute
        continue
    top_words, top_sims = find_knn_words_array(confusion[0], confusion[1][2], tokenizer, bert_model, top = 3)
    confusion_result = (confusion[0], confusion[1], top_words, top_sims)
    confusion_results.append(confusion_result)
    if (top_words[0]==confusion[1][0]): # most similar word matches original
        m_original += 1
    elif(top_words[0]==confusion[1][1]): # most similar word matches itself
        m_itself += 1
    else: # most similar word matches other word
        m_other += 1

print(f"m_itself: {m_itself}, m_original: {m_original}, m_other: {m_other}")

print("m_itself: ", round(m_itself/(m_itself+m_original+m_other),3))
print("m_original: ",round(m_original/(m_itself+m_original+m_other), 3))
print("m_other: ",round(m_other/(m_itself+m_original+m_other), 3))


m_itself = 0
m_other = 0
confusion_results = []
for train_idx in range(len(norne['train'])): 
    train_row = norwegian_ner['train'][train_idx]
    relevant_ids = find_relevant_word_ids(train_row['tokens'])
    if (not relevant_ids): # did not find any words to substitute
        continue
        
    # find most similar word to all the relevant words
    most_similar_words = find_knn_words_array_multi(train_row['tokens'], relevant_ids, tokenizer, bert_model, top = 1)
    
    for i, relevant_idx in enumerate(relevant_ids): 
        if (train_row['tokens'][relevant_idx].lower()==most_similar_words[i]): # most similar word matches original
            m_itself += 1
        else: # most similar word matches other word
            m_other += 1
            
print(f"m_itself: {m_itself}, m_other: {m_other}")

print("m_itself: ", round(m_itself/(m_itself+m_original+m_other),3))
print("m_other: ",round(m_other/(m_itself+m_original+m_other), 3))




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

# define cosine function 
cos = CosineSimilarity(dim=1)

def find_knn_words_array(word_array, word_idx, tokenizer, bert_model, layer = 12, top = 10): 
    word_embedding = torch.tensor(find_word_embedding_idx(word_idx, word_array, tokenizer, bert_model, LAYER = layer)).clone().to("cuda")
    
    return get_most_similar_from_vector(word_embedding, word_tensors_cuda, dict_words, is_cuda = True, top = top)


def find_relevant_word_ids(tokenized_chunk): 
    # extract indecies of relevant candidate words
    relevant_words = []
    for i, word in enumerate(tokenized_chunk): 
        if (word.lower() in confusion_sets): 
            relevant_words.append(i)
            
    return relevant_words

def find_knn_words_array_multi(word_array, relevant_ids, tokenizer, bert_model, layer = 12, top = 10):
    token_mapping, token_embeddings = tokens_to_embeddings(word_array, tokenizer, bert_model)
    
    most_similar_words = []
    for word_idx in relevant_ids: 
        # obtain embeddings for word
        token_ids = np.where(token_mapping == word_idx)[0]
        word_embedding = torch.tensor(np.mean(token_embeddings[token_ids], axis=0)[layer]).clone().to("cuda")
        # find most similar
        most_similar_word = get_most_similar_from_vector(word_embedding, word_tensors_cuda, dict_words, is_cuda = True, top = top)[0][0]
        most_similar_words.append(most_similar_word)
    
    return most_similar_words


#########################
# Load data
#########################

## Load confusion set
with open(SAVING_FOLDER+"words_with_confusions.json", "r") as f: 
    confusion_sets = json.load(f)

## Load in Treebank
norne = load_dataset('NbAiLab/norne', 'bokmaal')

## Load in treebank with confusions
with open(SAVING_FOLDER+"norne_with_confusions.json", "r") as f: 
    norne_train_confusions = json.load(f)
    
    

#########################
# Running program
#########################

wrong_context_results = []
for layer in range(13): 
    print(layer)
    # make relevant layer into cuda norwegian
    word_tensors_cuda = word_tensors[:, layer, :].clone().to("cuda")
    
    ### Words in wrong context
    m_itself = 0
    m_original = 0
    m_other = 0
    confusion_results = []
    for train_idx in range(len(norne_train_confusions)): 
        train_row = norne_train_confusions[train_idx]
        top_words, top_sims = find_knn_words_array(train_row[0], train_row[1][2], tokenizer, bert_model, layer = layer, top = 3)
        confusion_result = (train_row[0], train_row[1], top_words, top_sims)
        confusion_results.append(confusion_result)
        if (top_words[0]==train_row[1][0]): # most similar word matches original
            m_original += 1
        elif(top_words[0]==train_row[1][1]): # most similar word matches itself
            m_itself += 1
        else: # most similar word matches other word
            m_other += 1
    
    print(f"m_itself: {m_itself}, m_original: {m_original}, m_other: {m_other}")
    
    print("m_itself: ", round(m_itself/(m_itself+m_original+m_other),3))
    print("m_original: ",round(m_original/(m_itself+m_original+m_other), 3))
    print("m_other: ",round(m_other/(m_itself+m_original+m_other), 3))
    
    wrong_context_results.append((m_itself, m_original, m_other))



real_context_results = []
for layer in range(13): 
    print(layer)
    # make relevant layer into cuda norwegian
    word_tensors_cuda = word_tensors[:, layer, :].clone().to("cuda")
    
    m_itself = 0
    m_other = 0
    confusion_results = []
    for train_idx in range(len(norne['train'])): 
        train_row = norne['train'][train_idx]
        relevant_ids = find_relevant_word_ids(train_row['tokens'])
        if (not relevant_ids): # did not find any words to substitute
            continue
            
        # find most similar word to all the relevant words
        most_similar_words = find_knn_words_array_multi(train_row['tokens'], relevant_ids, tokenizer, bert_model, layer = layer,  top = 1)
        
        for i, relevant_idx in enumerate(relevant_ids): 
            if (train_row['tokens'][relevant_idx].lower()==most_similar_words[i]): # most similar word matches original
                m_itself += 1
            else: # most similar word matches other word
                m_other += 1
                
    print(f"m_itself: {m_itself}, m_other: {m_other}")
    
    print("m_itself: ", round(m_itself/(m_itself+m_original+m_other),3))
    print("m_other: ",round(m_other/(m_itself+m_original+m_other), 3))
    
    real_context_results.append((m_itself, m_other))
    
    
# save end results
with open(SAVING_FOLDER+"real_wrong_context_results.json", "w") as f: 
    json.dump({'wrong_context': wrong_context_results, 'real_context': real_context_results}, f)
    
print("Program finished")




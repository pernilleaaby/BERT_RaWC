# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:42:07 2022

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

from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertTokenizerFast, BertModel

import sys
sys.path.append("../utils/")

from model_utils import *
from load_dict import * 

from translate.storage.tmx import tmxfile



# define constants
DATASET_FOLDER = "D:/datasets/"
SAVING_FOLDER = "../data/"

mBERT = True
ALPHA = 1
if (mBERT): 
    WORD_FILE_NO = "../data/embeddings/word_to_embedding_mBERT.npy"
    EMBEDDING_FILE_NO = "../data/embeddings/average_embeddings_mBERT.npy"
    WORD_FILE_EN = "../data/embeddings/word_to_embedding_en_mBERT.npy"
    EMBEDDING_FILE_EN = "../data/embeddings/average_embeddings_en_mBERT.npy"
    CONTEXT_RESULT_FILE = "results_mt_context_en2no_mBERT_"+str(ALPHA)+".json"
    NON_CONTEXT_RESULT_FILE = "results_mt_non_context_en2no_mBERT_"+str(ALPHA)+".json"
else: 
    WORD_FILE_NO = "../data/embeddings/word_to_embedding.npy"
    EMBEDDING_FILE_NO = "../data/embeddings/average_embeddings.npy"
    WORD_FILE_EN = "../data/embeddings/word_to_embedding_en.npy"
    EMBEDDING_FILE_EN = "../data/embeddings/average_embeddings_en.npy"
    CONTEXT_RESULT_FILE = "results_mt_context_en2no_"+str(ALPHA)+".json"
    NON_CONTEXT_RESULT_FILE = "results_mt_non_context_en2no_"+str(ALPHA)+".json"
    
# choose random sentence pairs
SAMPLE_SIZE = 5000
    
    
## Load Data

word_tensors = torch.tensor(np.load(EMBEDDING_FILE_NO), dtype=torch.float32)
dict_words = np.load(WORD_FILE_NO, allow_pickle=True)

# load english word-embeddings
word_tensors_en = torch.tensor(np.load(EMBEDDING_FILE_EN), dtype=torch.float32)
dict_words_en = np.load(WORD_FILE_EN, allow_pickle=True)

# load bert-model
if (mBERT): 
    model_name = 'bert-base-multilingual-cased'
else: 
    model_name = 'NbAiLab/nb-bert-base' 
tokenizer, bert_model = load_bert(model_name, is_cuda = True)

# load parallel corpus Norwegian English
with open(SAVING_FOLDER+"riksrevisjonen.no.en-nb.tmx", 'rb') as fin:
        tmx_file = tmxfile(fin, 'no', 'en')
parallel_data = []
for node in tmx_file.unit_iter():
    parallel_data.append((node.source, node.target))

# load word translations
with open(SAVING_FOLDER+"en2no_test_dictionary.json", "r") as f: 
    en2no_dict = json.load(f)
    
# extract a set of sentences
#sample_sentences = random.sample(parallel_data, SAMPLE_SIZE)
sample_sentences = parallel_data[:SAMPLE_SIZE]


## Align to find word-pairs, source: English, target: Norwegian

def find_word_pairs(word_array_en, word_array_no): 
    word_pairs = []
    no_ids = []
    for i, en_word in enumerate(word_array_en):
        if (en_word not in en2no_dict): 
            continue
        else: 
            no_options = en2no_dict[en_word]
            word_pair = False
            for j, no_word in enumerate(word_array_no):
                if (no_word in no_options and not word_pair): 
                    word_pair = (i, j, en_word, no_word)
                elif (no_word in no_options): # found double match, so we remove pair
                    continue
            # add pair
            if (word_pair): 
                # make sure the norwegian word is not used twice
                if (word_pair[1] in no_ids): 
                    # remove the relevant word_pair
                    rel_idx = no_ids.index(word_pair[1])
                    del no_ids[rel_idx]
                    del word_pairs[rel_idx]
                else: 
                    word_pairs.append(word_pair)
                    no_ids.append(word_pair[1])
            
    return word_pairs

def language_adjust(embeddings, layer, from_language = 'en'): 
    # create language vector
    no_mean_vector = torch.mean(word_tensors[:, layer, :], 0).detach().numpy()
    en_mean_vector = torch.mean(word_tensors_en[:, layer, :], 0).detach().numpy()
    
    if (from_language == 'en'): 
        adjusted_embeddings = [embedding+ALPHA*(-en_mean_vector+no_mean_vector) for embedding in embeddings]
    else: # assume from norwegian
        adjusted_embeddings = [embedding+ALPHA*(-no_mean_vector+en_mean_vector) for embedding in embeddings]
        
    return adjusted_embeddings


aligned_en_no_data = []
for sent_tuple in sample_sentences: 
    word_array_en = nltk.word_tokenize(sent_tuple[0].lower())
    word_array_no = nltk.word_tokenize(sent_tuple[1].lower())
    
    word_pairs = find_word_pairs(word_array_en, word_array_no)
    
    if (word_pairs): # we only add the sentence if we find at least one word-pair
        aligned_en_no_data.append((word_array_en, word_array_no, word_pairs))
        
# Print some information about number of words
english_words_from_pairs = [word_pair[2] for word_pairs in aligned_en_no_data for word_pair in word_pairs[2]]
norwegian_words_from_pairs = [word_pair[3] for word_pairs in aligned_en_no_data for  word_pair in word_pairs[2]]

word_pairs_len = len(english_words_from_pairs)
u_en_words = len(set(english_words_from_pairs))
u_no_words = len(set(norwegian_words_from_pairs))
u_word_pairs = len(set(zip(english_words_from_pairs, norwegian_words_from_pairs)))
print(word_pairs_len, u_en_words, u_no_words, u_word_pairs)


## Create contextual embeddings for all the word-pairs
en_embeddings = []
no_embeddings = []
embedding_word_pairs = []

for i, aligned_words in enumerate(aligned_en_no_data): 
    words_en = aligned_words[0]
    words_no = aligned_words[1]
    word_pairs = aligned_words[2]
    
    token_mapping_en, token_embeddings_en = tokens_to_embeddings(words_en, tokenizer, bert_model)
    token_mapping_no, token_embeddings_no = tokens_to_embeddings(words_no, tokenizer, bert_model)
    
    for word_pair in word_pairs: 
        token_ids_en = np.where(token_mapping_en == word_pair[0])[0]
        word_embedding_en = np.mean(token_embeddings_en[token_ids_en], axis=0)
        
        token_ids_no = np.where(token_mapping_no == word_pair[1])[0]
        word_embedding_no = np.mean(token_embeddings_no[token_ids_no], axis=0)
        
        if (np.isnan(np.sum(word_embedding_en)) or np.isnan(np.sum(word_embedding_no))): # we skip the embedding in care of som error
            continue 
            
        en_embeddings.append(word_embedding_en)
        no_embeddings.append(word_embedding_no)
        embedding_word_pairs.append((i, word_pair))
    

## Evaluate word retrieval performance
results = []
cos = CosineSimilarity(dim = 1)

def match_context_embeddings(en_embeddings_l, no_embeddings_l): 
    # make embeddings to cuda
    en_embeddings_l = torch.tensor(en_embeddings_l).clone().to("cuda")
    # make relevant layer into cuda english
    no_embeddings_l = torch.tensor(no_embeddings_l).clone().to("cuda")
    
    corrects = 0
    
    for en_idx in range(en_embeddings_l.shape[0]):
        cos_sims = cos(en_embeddings_l[en_idx].unsqueeze(0), no_embeddings_l).cpu().detach().numpy()
        resulting_idx = np.argmax(cos_sims)
    
        if (en_idx == resulting_idx): 
            corrects += 1
            
    return corrects

for layer in range(13): 
    en_embeddings_l = np.array(en_embeddings)[:, layer, :]
    no_embeddings_l = np.array(no_embeddings)[:, layer, :]
    
    # run without adjusting for language
    corrects = match_context_embeddings(en_embeddings_l, no_embeddings_l)
    
    # run with adjusting for language
    en_embeddings_l_ad = language_adjust(en_embeddings_l, layer, from_language = 'en')
    
    corrects_adj = match_context_embeddings(en_embeddings_l_ad, no_embeddings_l)
    results.append((corrects/en_embeddings_l.shape[0], corrects_adj/en_embeddings_l.shape[0]))
    
    print(layer)
    print((corrects/en_embeddings_l.shape[0], corrects_adj/en_embeddings_l.shape[0]))
    
    
# save mt results
with open(SAVING_FOLDER+CONTEXT_RESULT_FILE, "w") as f: 
    json.dump(results, f)
    
    
    
    
## Create non-contextual experiment, only using unique word-pairs
# find relevant indecies
non_contextual_indecies = []
chosen_word_pairs = set()
i = 0
for word_pair in zip(english_words_from_pairs, norwegian_words_from_pairs): 
    if (word_pair not in chosen_word_pairs): 
        non_contextual_indecies.append(i)
        chosen_word_pairs.add(word_pair)
    i += 1
non_contextual_indecies = np.array(non_contextual_indecies)
print("We have this many unique word pairs: ", len(non_contextual_indecies))

# test all layers
results_non = []
for layer in range(13): 
    en_embeddings_l = np.array(en_embeddings)[non_contextual_indecies, layer, :]
    no_embeddings_l = np.array(no_embeddings)[non_contextual_indecies, layer, :]
    
    # run without adjusting for language
    corrects = match_context_embeddings(en_embeddings_l, no_embeddings_l)
    
    # run with adjusting for language
    en_embeddings_l_ad = language_adjust(en_embeddings_l, layer, from_language = 'en')
    
    corrects_adj = match_context_embeddings(en_embeddings_l_ad, no_embeddings_l)
    results_non.append((corrects/en_embeddings_l.shape[0], corrects_adj/en_embeddings_l.shape[0]))
    
    print(layer)
    print((corrects/en_embeddings_l.shape[0], corrects_adj/en_embeddings_l.shape[0]))



# save mt results
with open(SAVING_FOLDER+NON_CONTEXT_RESULT_FILE, "w") as f: 
    json.dump(results_non, f) 
    
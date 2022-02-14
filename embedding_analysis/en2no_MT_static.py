# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:30:45 2022

@author: perni
"""

from datetime import datetime
import re
import pandas as pd
import numpy as np
import math
import random
import json

import spacy

import torch
from torch.nn import CosineSimilarity

import sys
sys.path.append("../utils/")

from model_utils import *
from load_dict import * 

# define constants
DATASET_FOLDER = "D:/datasets/"
SAVING_FOLDER = "../data/"
mBERT = False
if (mBERT): 
    WORD_FILE_NO = "../data/embeddings/word_to_embedding_mBERT.npy"
    EMBEDDING_FILE_NO = "../data/embeddings/average_embeddings_mBERT.npy"
    WORD_FILE_EN = "../data/embeddings/word_to_embedding_en_mBERT.npy"
    EMBEDDING_FILE_EN = "../data/embeddings/average_embeddings_en_mBERT.npy"
    MT_RESULT_FILE = "results_mt_static_en2no_mBERT.json"
    MT_RESULT_FILE_ADJUSTED = "results_mt_static_en2no_adjusted_mBERT.json"
else: 
    WORD_FILE_NO = "../data/embeddings/word_to_embedding.npy"
    EMBEDDING_FILE_NO = "../data/embeddings/average_embeddings.npy"
    WORD_FILE_EN = "../data/embeddings/word_to_embedding_en.npy"
    EMBEDDING_FILE_EN = "../data/embeddings/average_embeddings_en.npy"
    MT_RESULT_FILE = "results_mt_static_en2no.json"
    MT_RESULT_FILE_ADJUSTED = "results_mt_static_en2no_adjusted.json"
    
###########################
## Load data
###########################

# load norwegian word-embeddings
word_tensors = torch.tensor(np.load(EMBEDDING_FILE_NO), dtype=torch.float32)
dict_words = np.load(WORD_FILE_NO, allow_pickle=True)

# load english word-embeddings
word_tensors_en = torch.tensor(np.load(EMBEDDING_FILE_EN), dtype=torch.float32)
dict_words_en = np.load(WORD_FILE_EN, allow_pickle=True)

# load word translations
with open(SAVING_FOLDER+"en2no_test_dictionary.json", "r") as f: 
    en2no_dict = json.load(f)
    
    
###########################
## Test translations
###########################

def test_translations(layer, language_adjust = False): 
    # make relevant layer into cuda norwegian
    word_tensors_cuda = word_tensors[:, layer, :].clone().to("cuda")
    # make relevant layer into cuda english
    word_tensors_en_cuda = word_tensors_en[:, layer, :].clone().to("cuda")
    
    # create language vector
    no_mean_vector = torch.mean(word_tensors_cuda, 0)
    en_mean_vector = torch.mean(word_tensors_en_cuda, 0)
    
    at_top1 = 0
    at_top3 = 0
    at_top10 = 0

    i = 0
    for i, word_en in enumerate(dict_words_en): 
        word_vector_en = word_tensors_en_cuda[i]
        if (language_adjust): # do additional language adjustment
            word_vector_en = word_vector_en-en_mean_vector+no_mean_vector
        # find top matches
        top_words, top_sims = get_most_similar_from_vector(word_vector_en, word_tensors_cuda, dict_words, is_cuda = True, top = 10)

        if (top_words[0] in en2no_dict[word_en]): 
            at_top1 += 1
            at_top3 += 1
            at_top10 += 1
        elif (any([top_word in en2no_dict[word_en] for top_word in top_words[:3]])): 
            at_top3 += 1
            at_top10 += 1
        elif (any([top_word in en2no_dict[word_en] for top_word in top_words[:10]])): 
            at_top10 += 1
        #else: 
        #    print(word_en)
        #    print(en2no_dict[word_en])
        #    print(top_words)
            


        i += 1
        
    return at_top1, at_top3, at_top10

mt_results = []
for layer in range(13): 
    print("Testing layer: ", layer)
    at_top1, at_top3, at_top10 = test_translations(layer)
    print(at_top1, at_top3, at_top10)
    print("Top1", round(at_top1/len(dict_words_en), 3))
    print("Top3", round(at_top3/len(dict_words_en), 3))
    print("Top10", round(at_top10/len(dict_words_en), 3))
    mt_results.append((at_top1, at_top3, at_top10))
    
    
# Save results
with open(SAVING_FOLDER+MT_RESULT_FILE, "w") as f: 
    json.dump(mt_results, f)
    
    
# Test translation with adjusting
mt_results_adjusted = []
for layer in range(13): 
    print("Testing layer: ", layer)
    at_top1, at_top3, at_top10 = test_translations(layer, language_adjust = True)
    print(at_top1, at_top3, at_top10)
    print("Top1", round(at_top1/len(dict_words_en), 3))
    print("Top3", round(at_top3/len(dict_words_en), 3))
    print("Top10", round(at_top10/len(dict_words_en), 3))
    mt_results_adjusted.append((at_top1, at_top3, at_top10))
    
# save mt results
with open(SAVING_FOLDER+MT_RESULT_FILE_ADJUSTED, "w") as f: 
    json.dump(mt_results_adjusted, f)
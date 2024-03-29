# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:26:50 2022

@author: perni
"""

import pandas as pd
import numpy as np
import json

import spacy
from nltk.corpus import brown

import sys
sys.path.append("../utils/")

from model_utils import * 

# define constants
DATA_FOLDER = "../data/"
SAVING_FOLDER = "../data/embeddings/"

MODEL_NAME = 'bert-base-multilingual-cased'#'NbAiLab/nb-bert-base'
WORD_FILE = "word_to_embedding_en_mBERT.npy"
EMBEDDING_FILE = "average_embeddings_en_mBERT.npy"

# load bert-model
tokenizer, bert_model = load_bert(MODEL_NAME, is_cuda = True)


##############################################################
# Load data to use as inference corpus to obtain embeddings
##############################################################

# load words we need to translate from 
with open(DATA_FOLDER+"en2no_test_dictionary.json", "r") as f: 
    en2no_dict = json.load(f)

relevant_words = [word for word in en2no_dict.keys()]


## Load brown corpus
brown_paras = brown.paras()
# make set of two and two sentences
brown_sents = []
for para in brown_paras: 
    for i in range(len(para)): 
        if (i % 2 == 0): 
            brown_sents.append(para[i])
        else: 
            brown_sents[-1] += para[i]
#brown_sents = [sent for para in brown_paras for sent in para]

print("Number  of text chunks", len(brown_sents), flush=True)


##############################################################
# Create averaged embeddings
##############################################################

def save_word_embeddings(word_embedding_dict): 
    fullform_words = list(word_embedding_dict.keys())
    word_tensors = [word_embedding_dict[key][0] for key in fullform_words]
    
    with open(SAVING_FOLDER+WORD_FILE, "wb") as f: 
        np.save(f, np.array(fullform_words, dtype=object))
    with open(SAVING_FOLDER+EMBEDDING_FILE, "wb") as f: 
        np.save(f, np.array(word_tensors, dtype=np.float32))
        
    print("Embeddings saved", flush=True)
    
word_embedding_dict = {}

for j, word_array in enumerate(brown_sents): 
    token_mapping, token_embeddings = tokens_to_embeddings(word_array, tokenizer, bert_model)

    for i, word in enumerate(word_array): 
        word = word.lower()
        if (word in relevant_words):
            # obtain embeddings for word
            token_ids = np.where(token_mapping == i)[0]
            word_embedding = np.mean(token_embeddings[token_ids], axis=0)
            # make sure word embedding does not have nan, for some reason it happens sometimes
            if (np.isnan(np.sum(word_embedding))): 
                continue
            # average in static word-embeddings
            word_embedding_dict = insert_embedding_in_dict(word, word_embedding, word_embedding_dict)

    if (j % 1000 == 0):
        print(j, end=" ", flush=True)

# save word embeddings when finished
save_word_embeddings(word_embedding_dict)
print("Program finished.")
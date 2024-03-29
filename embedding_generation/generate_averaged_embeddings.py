# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:08:21 2022

@author: perni
"""
import re

import pandas as pd
import numpy as np
import json

import spacy

import sys
sys.path.append("../utils/")

from model_utils import * 

# define constants
DATA_FOLDER = "../data/"
SAVING_FOLDER = "../data/embeddings/"

MODEL_NAME = 'bert-base-multilingual-cased'#'NbAiLab/nb-bert-base'
WORD_FILE = "word_to_embedding_mBERT.npy"
EMBEDDING_FILE = "average_embeddings_mBERT.npy"

# load bert-model
tokenizer, bert_model = load_bert(MODEL_NAME, is_cuda = True)

# load tokenizer
nlp = spacy.load("nb_core_news_sm", disable = ['ner', 'tagger', 'parser', 'lemmatizer', 'tok2vec', 'morphologizer', 'attribute_ruler'])


##############################################################
# Load data to use as inference corpus to obtain embeddings
##############################################################

# load filtered_words, words in both dictionary and the news corpus
with open(DATA_FOLDER+"top50000_news_vocabulary.json", "r") as f: 
    filtered_words_occ = json.load(f)

relevant_words = [word[0] for word in filtered_words_occ]


## Code for chunk partition
text_chunks = []
with open(DATA_FOLDER+"subset_newscorpus.txt", "r", encoding="UTF-8") as f_in: 
    for i, chunk in enumerate(f_in): 
        text_chunks.append(chunk)

print("Number  of text chunks", len(text_chunks), flush=True)


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

for j, chunk in enumerate(text_chunks): 
    chunk = chunk.strip()
    word_array = [token.text for token in nlp(chunk)]
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

    if (j % 10000 == 0):
        print(j, end=" ", flush=True)
    
    # do intermediate saving
    if ((j+1) % 100000 == 0): 
        save_word_embeddings(word_embedding_dict)

# save word embeddings when finished
save_word_embeddings(word_embedding_dict)
print("Program finished.")

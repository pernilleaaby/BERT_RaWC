# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:55:13 2022

@author: perni
"""

import re

import pandas as pd
import numpy as np

from collections import Counter
import pickle

import json
from datetime import datetime
import spacy

import sys
sys.path.append("../utils/")


# constants
DATA_FOLDER = "D:/datasets/"
SAVING_FOLDER = "../data/"
TOP_COMMON = 50000

# load spacy tokenizer
nlp = spacy.load("nb_core_news_sm", disable = ['ner', 'tagger', 'parser', 'lemmatizer', 'tok2vec', 'morphologizer', 'attribute_ruler'])


# count the words in the corpus in lowercase
cnt_words = Counter()
word_array = []
with open(DATA_FOLDER+"cleaned_newspapers_online_nob.txt", "r", encoding="UTF-8") as f_in: 
    articles = []
    for i, article in enumerate(f_in): 
        articles.append(article)
         
        if (i % 10000 == 0): # process
            docs = nlp.tokenizer.pipe(articles)
            for doc in docs: 
                # add words from article
                word_array += [token.text.lower() for token in doc]
            print(i, end=" ")
            
            # add object to counter
            cnt_words += Counter(word_array)
            word_array = []
            articles = []
            

## Filter out words not in the norwegian dictionary
from load_dict import * 

# load word set
phonetics_df = load_phonetic_frame(DATA_FOLDER+"20191016_nlb_trans/nlb_nob_20181129.lex")
dictionary_word_set = set([word.lower() for word in phonetics_df[0]])

filtered_words_occ = []
for word_tuple in cnt_words.most_common(): 
    if (word_tuple[0] in dictionary_word_set):
        filtered_words_occ.append(word_tuple)
filtered_words_occ = filtered_words_occ[:TOP_COMMON]

# save word vocabulary
with open(SAVING_FOLDER+"top"+str(TOP_COMMON)+"_news_vocabulary.json", "w") as f: 
    json.dump(filtered_words_occ, f)
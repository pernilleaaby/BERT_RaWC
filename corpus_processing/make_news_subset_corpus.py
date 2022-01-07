# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:57:51 2022

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
import spacy_fastlang
import nltk


# constants
DATA_FOLDER = "D:/datasets/"
SAVING_FOLDER = "../data/"
WORD_MIN = 50

# load spacy tokenizer
nlp = spacy.load("nb_core_news_sm", disable = ['ner', 'tagger', 'parser', 'lemmatizer', 'tok2vec', 'morphologizer', 'attribute_ruler'])
nlp.add_pipe('language_detector')

# load word vocabulary 
with open(SAVING_FOLDER+"top50000_news_vocabulary.json", "r") as f: 
    filtered_words_occ = json.load(f)
    
filtering_count_dict = dict()
for word_tuple in filtered_words_occ:  
    filtering_count_dict.update({word_tuple[0]: 0})
        
word_vocabulary = set(filtering_count_dict.keys())

## create subcorpus with at least word-min occurances

def article2text_chunks(article): 
    text_chunks = []
    doc = nltk.sent_tokenize(article)
    for i in range(0, len(doc), 2): 
        text_chunks.append(" ".join(doc[i:i+2]))
    
    return text_chunks

sub_chunks = []

with open(DATA_FOLDER+"cleaned_newspapers_online_nob.txt", "r", encoding="UTF-8") as f_in: 
    articles = []
    for i, article in enumerate(f_in): 
        # print progress
        if (i % 1000 == 0): 
            print(i, len(word_vocabulary), end=" ")
            
        text_chunks = article2text_chunks(article)
        for chunk in text_chunks:
            doc = nlp(chunk)
            if (doc._.language == 'en'): # we want to filter out english text
                continue
            
            # add words from article
            words_in_text = set([token.text.lower() for token in doc])
            
            common_words = list(words_in_text.intersection(word_vocabulary))
            
            # add text to subcorpus if there exists common words
            if (common_words != []): 
                sub_chunks.append(chunk)
            
                # add count (or remove word elements) from filtering set
                for word in common_words:
                    if (word in filtering_count_dict): 
                        new_count = filtering_count_dict[word] + 1
                        if (new_count >= WORD_MIN): # we remove word from filtering list
                            word_vocabulary.remove(word)
                        else: 
                            filtering_count_dict[word] = new_count
                            
                    
        if (len(word_vocabulary)== 0): 
            break
        
## Save subcorpus
with open(SAVING_FOLDER+"subset_newscorpus_min"+str(WORD_MIN)+".txt", "w", encoding="UTF-8") as f: 
    for chunk in sub_chunks: 
        f.write(chunk + "\n")
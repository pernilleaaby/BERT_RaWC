# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:34:31 2022

@author: perni
"""

## Create a dictionary that maps english words to all its norwegian translations 
# from the MUSE benchmark

import re
import json 

import pandas as pd
import requests

# constants
DATA_FOLDER = "D:/datasets/"
SAVING_FOLDER = "../data/"

## Load data

with open(DATA_FOLDER+"MT/no-en_words.txt", "r", encoding = "utf-8") as f: #
    no3en_muse = f.read()
    no3en_muse = [line.split("\t") for line in no3en_muse.split("\n")]
    

no3en_muse_df = pd.DataFrame(no3en_muse)

# do encoding formatting
new_no = []
for word in no3en_muse_df[0]: 
    try: 
        new_no.append(word.encode('latin1').decode('utf8'))
    except: 
        #print(word)
        new_no.append(word)
        
new_en = []
for word in no3en_muse_df[1]: 
    try: 
        new_en.append(word.encode('latin1').decode('utf8'))
    except: 
        #print(word)
        new_en.append(word)

no3en_muse_df['no'] = new_no
no3en_muse_df['en'] = new_en


# load norwegian embedding vocabulary
with open(SAVING_FOLDER+"top50000_news_vocabulary.json", "r") as f: 
    embedding_vocab = json.load(f) 
    embedding_vocab = dict(embedding_vocab)

in_embedding_vocab = [no_word in embedding_vocab for no_word in no3en_muse_df['no']]

muse_filtered_df = no3en_muse_df.loc[in_embedding_vocab]

muse_filtered_df.shape, no3en_muse_df.shape

## Make en -> no word translation dictionary

english_words = list(muse_filtered_df.en)
u_english_words = set(muse_filtered_df.en)

# we see that there are multiple one-many relationships
len(english_words), len(u_english_words)

en2no_dict = dict()
for no, en in zip(muse_filtered_df.no, muse_filtered_df.en): 
    if (en in en2no_dict): 
        en2no_dict[en].append(no)
    else: 
        en2no_dict.update({en: [no]})

print(len(en2no_dict))


## Save translation-set

with open(SAVING_FOLDER+"en2no_test_dictionary.json", "w") as f: 
    json.dump(en2no_dict, f)
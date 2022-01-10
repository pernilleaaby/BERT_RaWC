# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:04:38 2022

@author: perni
"""

# we use jellyfish as it also has damerau distance 
import jellyfish

import pandas as pd
import numpy as np
import json
from collections import Counter

import sys
sys.path.append("../utils/")

from load_dict import * 
from levensthein_utils import * 


# define constants
DATASET_FOLDER = "D:/datasets/"
SAVING_FOLDER = "../data/"


# load dictionary with phonetic encoding
phonetics_df = load_phonetic_frame(DATASET_FOLDER+"20191016_nlb_trans/nlb_nob_20181129.lex")

# load our top 50 000 vocabulary from news corpus
with open(SAVING_FOLDER+"top50000_news_vocabulary.json", "r") as f: 
    filtered_words_occ = json.load(f)
# extract only the words 
filtered_word_set = set([word_tuple[0] for word_tuple in filtered_words_occ])


# create a subset of the phonetics frame which only include the top 50 000 vocabulary
word_list = [word.lower() for word in phonetics_df[0]]
in_filtered_words = [word in filtered_word_set for word in word_list]
relevant_word_df = phonetics_df.loc[in_filtered_words]        
        
# zip all the relevant tuples from dataframe
fullform_words = [word.lower() for word in relevant_word_df[0]]
phon_encodings = list(relevant_word_df['lev_adopted_encoding'])

zipped_word_phon = list(zip(fullform_words, phon_encodings))

# find confusion candidates
words_substitute_indecies = []
words_edit_distances = []
for i, word_tuple in enumerate(zipped_word_phon): 
    substitute_indecies, edit_distances = find_substitute_words(word_tuple, zipped_word_phon, candidates = 100, max_dist = 3)
    words_substitute_indecies.append(substitute_indecies)
    words_edit_distances.append(edit_distances)
    
    if (i % 100 == 0): 
        print(i, end = " ")
    
# end saving
with open(SAVING_FOLDER+"phonetic_damerau_substitutes.npy", "wb") as f: 
    np.save(f, np.array(words_substitute_indecies, dtype=object))
with open(SAVING_FOLDER+"phonetic_damerau_edit_distance.npy", "wb") as f: 
    np.save(f, np.array(words_edit_distances, dtype=object))

print("Finished generating levenshtein alternatives for phonetics. ", flush=True)


# create a simple word to confusion candidate dictionary
word2confusions = dict()
max_len = 10
for i, word_tuple in enumerate(zipped_word_phon): 
    substitute_indecies = words_substitute_indecies[i]
    sub_list = []
    for sub_idx in substitute_indecies: 
        sub_word = zipped_word_phon[sub_idx][0]
        if (sub_word in sub_list): # drop adding same word more than once
            continue
        sub_list.append(sub_word)
        if (len(sub_list) >= max_len): 
            break
    # add word and confusions to dictionary
    if (sub_list): 
        word2confusions.update({word_tuple[0]: sub_list})
    else: 
        print("Empty sub list", word_tuple)
        
# save simple confusion candidate list
with open(SAVING_FOLDER+"words_with_confusions.json", "w") as f: 
    json.dump(word2confusions, f)
    








# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:39:40 2022

@author: perni
"""

import json 
from datasets import load_dataset
import random

import sys
sys.path.append("../utils/")

from model_utils import *

SAVING_FOLDER = "../data/"

## Confusion function

def confuse_tokens(tokenized_chunk): 
    """Substitute one of the word tokens with a confusion candidate

    Parameters
    ----------
    tokenized_chunk : array
        index of relevant word in the word-array

    Returns
    -------
    array
        new tokenized_chunk with a confusion substitution 
    tuple 
        which word tokens was subsituted and the position in the 
        tokenized array
    """
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

## Load norne and add confusion

## Load in Treebank
norne = load_dataset('NbAiLab/norne', 'bokmaal')

## Load confusion set, necessary for soncusion generation
with open(SAVING_FOLDER+"words_with_confusions.json", "r") as f: 
    confusion_sets = json.load(f)

train_row = norne['train'][6]
confusion = confuse_tokens(train_row['tokens'])


norne_train_confusions = []
for train_row in norne['train']: 
    confusion = confuse_tokens(train_row['tokens'])
    if (not confusion): # does not find words to substitute
        continue
    norne_train_confusions.append((confusion[0], confusion[1], train_row['pos_tags']))

## Save the corpus with confused words

with open(SAVING_FOLDER+"norne_with_confusions.json", "w") as f: 
    json.dump(norne_train_confusions, f)

## Load corpus to inspect examples

with open(SAVING_FOLDER+"norne_with_confusions.json", "r") as f: 
    norne_train_confusions = json.load(f)
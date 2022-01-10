# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:05:29 2022

@author: perni
"""

# we use jellyfish as it also has damerau distance 
import jellyfish

import pandas as pd
import numpy as np
from collections import Counter


# function to find damerau levensthein distance between two words
def levenshtein_adopted_check(word_tuple, check_tuple): 
    # we filter word spelled the same and words with the same lemma
    if (word_tuple[0] == check_tuple[0]): 
        return 100
    else: 
        letter_dist = jellyfish.damerau_levenshtein_distance(word_tuple[0], check_tuple[0])
        phonetic_dist = jellyfish.damerau_levenshtein_distance(word_tuple[1], check_tuple[1])
        
        if (phonetic_dist < letter_dist): 
            return phonetic_dist + 0.5
        else: 
            return letter_dist

def levenshtein_multiple(word_tuple, word_dictionary): 
    lev_distances = list(map(levenshtein_adopted_check, [word_tuple]*len(word_dictionary), word_dictionary))
    
    return lev_distances
        
# find closest words with damerau-levensthein distance
def find_substitute_words(word_tuple, word_dictionary, candidates = 100, max_dist = 3): 
    
    # find distances and sort
    lev_distances = np.array(levenshtein_multiple(word_tuple, word_dictionary))
    sorted_distances = np.argsort(lev_distances)

    # only include words with max edits from the top candidates
    substitute_indecies = sorted_distances[:candidates][lev_distances[sorted_distances[:candidates]] <= max_dist]
    edit_distances = lev_distances[sorted_distances[:len(substitute_indecies)]]
    
    return  substitute_indecies, edit_distances
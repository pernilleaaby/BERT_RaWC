# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:07:47 2022

@author: perni
"""

import numpy as np 
import pandas as pd

import re
import glob



def load_phonetic_frame(input_file): 
    """Loads in a dictionary with phonetic encoding into a dataframe 
    with only the necessary data

    Parameters
    ----------
    input_file : str
        File path to the .txt file for the phonetic dictionary

    Returns
    -------
    dataframe
        pandas dataframe with relevant coulumns
    """
    with open(input_file, "r", encoding = 'iso-8859-1') as f: 
        text_data = f.read()
    
    splitted_lines = [line.split("\t") for line in text_data.split("\n")]

    # remove unecessary columns and NaNs
    phonetics_df = pd.DataFrame(splitted_lines).drop(columns = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15])
    phonetics_df = phonetics_df.dropna()


    def filter_stress(phonetic): 
        # functions to remove stress characters from SAMPA alphabet
        phonetic = re.sub("[\"%:\`']", "", phonetic)
        phonetic = re.sub("\$\s", "", phonetic)
        return phonetic

    def filter_lemma(lemma_long): 
        # format the lemma column from the file to only include lemma
        lemma = re.sub("lemma:","", lemma_long)
        lemma = lemma.split("|")[0]#re.sub("\|\d+", "", lemma)
        return lemma

    # create column with simplifies phonetic encoding and clean lemma column
    phonetics_df['phonetic_stressless'] = [filter_stress(phonetic) for phonetic in phonetics_df[1]] 
    phonetics_df['lemma'] = [filter_lemma(lemma_long) for lemma_long in phonetics_df[13]]
    
    
    # Additional formatting to make the phonetic encoding work with 
    # Levensthein distance, mainly double chars to single char
    
    # find all unique characters in phonetic alphabeth
    phonetics_list = list(phonetics_df['phonetic_stressless'])
    phonetic_chars = []
    for phonetic in phonetics_list: 
        phonetic_chars += phonetic.split()
    phonetic_chars = list(set(phonetic_chars))
    
    # define random characters not in phonestics to translate for phonetic single
    random_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'
    extra_set = []
    for char in random_chars: 
        if (not char in phonetic_chars):
            extra_set.append(char)
            
    # find all the double phonetic characters
    double_phonetic_chars = []
    for phonetic_char in phonetic_chars: 
        if (len(phonetic_char)>1): 
            double_phonetic_chars.append(phonetic_char)
    
    # define conversion from double char to random char not used as encoding before
    double2single = {}
    for i in range(len(double_phonetic_chars)): 
        double2single.update({double_phonetic_chars[i]: extra_set[i]})
    
    def phonetic_transformation(phonetic): 
        phonetic_splitted = phonetic.split()
        for i, phonetic_char in enumerate(phonetic_splitted): 
            if (phonetic_char in double2single): 
                phonetic_splitted[i] = double2single[phonetic_char]
        return "".join(phonetic_splitted)
    
    # create new list of phonetic list with substituted characters and removed spaces
    phonetics_df['lev_adopted_encoding'] = [phonetic_transformation(phonetic) for phonetic in phonetics_list]
    

    return phonetics_df
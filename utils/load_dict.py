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
    
    # remove rows without lemma
    phonetic_lemmas = np.array(phonetics_df.lemma)
    phonetics_df = phonetics_df.drop(np.where(phonetic_lemmas=='')[0])

    return phonetics_df
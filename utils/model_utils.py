# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 10:49:56 2022

@author: perni
"""

import pandas as pd
import numpy as np
import json

import torch
from transformers import BertTokenizerFast, BertModel
from torch.nn import CosineSimilarity

# define cosine function 
cos = CosineSimilarity(dim=1)


def load_bert(model, is_cuda = True):
    """Load/ download BERT model

    Parameters
    ----------
    model : str
        Name of spesific BERT model to download from huggingface
    is_cude: bool
        True if the script should run with cuda cores, cpu False

    Returns
    -------
    tokenizer
        Tokenizer to the given model
    bert_model
        Pre-trained BERT model
    """ 
    tokenizer = BertTokenizerFast.from_pretrained(model)
    bert_model = BertModel.from_pretrained(model)
    # change to cuda cores
    if (is_cuda): 
        bert_model.to('cuda')

    return tokenizer, bert_model


def tokens_to_embeddings(word_tokens, tokenizer, bert_model): 
    """Obtain token embeddings and token to word token mapping 
    from BERT-model

    Parameters
    ----------
    sent : array
        Input array of word tokens

    Returns
    -------
    list
        token to word id mapping
    list
        token embeddings from all layers
    """
    with torch.no_grad(): 
        inputs = tokenizer(word_tokens, return_tensors = "pt", truncation = True, max_length = 512, is_split_into_words=True)
        word_ids = inputs.word_ids(batch_index=0)
        outputs = bert_model(**inputs.to('cuda'), output_hidden_states=True)
        hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0) #stack all hidden states into same tensor
        token_embeddings = token_embeddings.squeeze(dim=1) # remove empty dimension
        token_embeddings = token_embeddings.permute(1,0,2)
        
    # convert to numpy arrays
    word_ids = np.array(word_ids)
    token_embeddings = token_embeddings.to("cpu").detach().numpy()
    
    return word_ids, token_embeddings 


def insert_embedding_in_dict(word_phrase, word_embedding, word_embedding_dict, max_count = 500): 
    """Insert a new word-embedding to the dictionary of embeddings, 
    add to mean if the word exists, or create new key.
    keeping track of averaged word-embeddings

    Parameters
    ----------
    word_phrase : str
        word
    word_embedding: array
        new word embedding for the given word
    word_embedding_dict: dict
        dicitonary with every word and embedding pair

    Returns
    -------
    dict
        dicitonary with word as key and word-embedding and appearances 
    """
    if (word_phrase in word_embedding_dict): 
        old_embedding = word_embedding_dict[word_phrase][0]
        old_occurances = word_embedding_dict[word_phrase][1] 

        # we do not update if word already contain more than 500 occurances
        if (old_occurances > max_count): 
            return word_embedding_dict
        
        # here we use "mean-pooling" over all the contextual embeddings 
        new_embedding = word_embedding/(old_occurances+1) + (old_embedding*old_occurances)/(old_occurances+1)
        word_embedding_dict[word_phrase] = (new_embedding, old_occurances+1)
    else: 
        word_embedding_dict.update({word_phrase: (word_embedding, 1)})
        
    return word_embedding_dict


###################################
# Processing embedding dictionary
###################################

def find_word_embedding_idx(word_idx, word_array, tokenizer, bert_model, LAYER = 12): 
    """Obtain and extract the word embeding of only the word-idx place
    in the word array. Layer is set to decide with hidden state to use.

    Parameters
    ----------
    word_idx : int
        index of relevant word in the word-array
    word_array: array
        array of words after word-tokenizing string
    LAYER: int
        integer to decide which layer to extract hidden state from

    Returns
    -------
    array
        vector corresponding to the embedding of the relevant word 
    """
    token_mapping, token_embeddings = tokens_to_embeddings(word_array, tokenizer, bert_model)
    
    # obtain embeddings for word
    token_ids = np.where(token_mapping == word_idx)[0]
    word_embedding = np.mean(token_embeddings[token_ids], axis=0)[LAYER]
    
    return word_embedding




def get_most_similar_from_vector(word_vector, word_tensors, dict_words, is_cuda = False, top = 10):
    """Find the most similar words to the input word-vector

    Parameters
    ----------
    word_vector : array
        embedding of a word
    word_tensor: tensor
        dictionary of embeddings
    dict_words: array
        array of words corresponding to the dictionary of embeddings

    Returns
    -------
    array
        top most similar words
    array
        the similarity score to the top words
    """
    # format word vector
    word_vector = word_vector.unsqueeze(0)
    # fetch cosine similarities
    if (is_cuda): 
        cos_sims = cos(word_vector, word_tensors).cpu().detach().numpy()
    else: 
        cos_sims = cos(word_vector, word_tensors).detach().numpy()
    sorted_indecies = np.argsort(cos_sims)
    
    return dict_words[sorted_indecies[-top:]][::-1], cos_sims[sorted_indecies][-top:][::-1]





# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:41:13 2017

@author: mallabi
"""
from nltk.text import Text
from modules import text_normalization as norm
from modules import ngram_analysis as ngm


def create_corpus(dataframe):
    corpus = dataframe.apply(lambda x: " ".join(x)).str.cat(sep=' ')
    corpus_list = dataframe.apply(lambda x: " ".join(x)).tolist() 
    return corpus, corpus_list

def get_context(term, series_tokens, series_raw=()):
    cleaned = Text(ngm.flatten_list(series_tokens))
    print ("Context in Cleaned Corpus")
    print(cleaned.concordance(term))
    if len(series_raw) !=0:    
        original = Text(ngm.flatten_list(series_raw.apply(norm.tokenize_text)))
        print("-------------------------------------------------------------------")
        print ("Context in Raw Corpus")
        print(original.concordance(term))
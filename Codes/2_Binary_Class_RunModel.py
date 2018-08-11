# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 14:36:08 2018

@author: bm30785
"""

#-------------------------------------------------------------------------------------------------------------
#DEFINE WORKING FOLDER and IMPORT PACKAGES
#-------------------------------------------------------------------------------------------------------------

import os
os.chdir("/home/bm30785/Codes")

#Import packages and dependencies
import nltk
import pickle
import pandas as pd
import numpy as np
from time import time
import re
import random

from modules import text_exploration as tx_exp
from modules import classify_models as skclassify

# Program start time
t_int = time()

#-------------------------------------------------------------------------------------------------------------
#SET INPUT and OUTPUT FOLDER PATHS
#-------------------------------------------------------------------------------------------------------------
#Set folder path
trans_data_path = "/home/bm30785/Data/Transcripts/Nexidia_DEU_Export_UAT/TranscriptData"
trans_data_path2 = "/home/bm30785/Data/Transcripts/Nexidia_DEU_Export_UAT/TranscriptData/Others"
interact_data_path="/home/bm30785/Data/Metadata"
output_data_path = "/home/bm30785/Data/PK_Data"

#-------------------------------------------------------------------------------------------------------------
#LOAD DATASETS
#-------------------------------------------------------------------------------------------------------------
t = time()
df_merged = pd.read_pickle(output_data_path + "/merged_binclass_data_full.pk")
print("Dataframe loaded in %0.3fs."% (time() - t))

#-------------------------------------------------------------------------------------------------------------
#FEATURE CREATION
#-------------------------------------------------------------------------------------------------------------
print("Create features set, flagging for complaint vs non-complaint")
#Splitting data into complaint and non-complaint subsets
#df_series = df_merged["Clean_Script"]
#df_series_comp = df_merged[df_merged.Complaints_ID=="Complaint"]["Clean_Script"]
#df_series_noncomp = df_merged[df_merged.Complaints_ID=="Non-Complaint"]["Clean_Script"]
print(df_merged.head())

features = df_merged.Clean_Script
labels= df_merged.Complaints_ID.apply(lambda x: 1 if x == "Complaint" else 0)

#Create Corpus
features_corpus,features_corpus_list = tx_exp.create_corpus(features)

print("Features in the sample")
print(features.head())

#-------------------------------------------------------------------------------------------------------------
#MODEL PREPARATION
#-------------------------------------------------------------------------------------------------------------
print("Create training and test datasets")
#Creating Training and Test Splits
train_corpus, test_corpus, train_labels, test_labels = skclassify.prepare_datasets(features_corpus_list, labels,proportion=0.3)

#-------------------------------------------------------------------------------------------------------------
#RUN MODELS
#-------------------------------------------------------------------------------------------------------------

print("Model runs")
#print(skclassify.inspect_LSA(train_corpus))

print("-"*90)
print("Limiting to 500 Features, unigram and bigrams model")
print("-"*90)
skclass_model = skclassify.skclassify_models(features_corpus_list,labels, ngram_range = (1,2), num_features = 500, lsa_features = 200)

print("-"*90)
print("Limiting to 500 Features, uni-gram model")
print("-"*90)
skclass_model = skclassify.skclassify_models(features_corpus_list,labels, ngram_range = (1,1), num_features = 500, lsa_features = 200)

print("Program completed in %0.3fs." % (time() - t_int))







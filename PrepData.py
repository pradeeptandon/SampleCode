# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 14:36:08 2018

@author: Pradeep
"""
#-------------------------------------------------------------------------------------------------------------
#DEFINE WORKING FOLDER and IMPORT PACKAGES
#-------------------------------------------------------------------------------------------------------------
import os
os.chdir("/home/")

import pandas as pd #used for dataframes
import numpy as np
import re,nltk,os,pprint,timeit
from nltk.stem.porter import *
from nltk.corpus import stopwords#,brown, treebank #Used for language processing
from sklearn import naive_bayes,svm
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve, auc
from nltk.tag import StanfordNERTagger
from ggplot import *
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from compiler.ast import flatten

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint
from sklearn.decomposition import LatentDirichletAllocation
import itertools

# Plotting tools
#import pyLDAvis
#import pyLDAvis.sklearn
import matplotlib.pyplot as plt
#Import custom dictionary

#Import custom dictionary


# Program start time
t_int = time()

#-------------------------------------------------------------------------------------------------------------
#SET INPUT and OUTPUT FOLDER PATHS
# Change the folder paths here for transcripts folder, metadata and output folder
#-------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------
#IMPORT DATASETS
#-------------------------------------------------------------------------------------------------------------
# Importing (1) Transcription data from zip files and (2) Interaction Metadata
print("Importing transcript data files")
df_trans_raw = pd.read_excel('abc.xlsx',sheet_name = 'sheet1') #Currently limited to 20 files for testing purposes

#Custom Dictionary
add_stoplist = {}
print("Removing Stopwords")
add_stoplist={"ok", "thank","much", "gon", "wan", "go", "yes","madam", "like","na", "ahead", 
"wan", "let", "u", "would","one","moment", "make", "sure", "dollar", "name","please",
"zero", "one", "two", "three", "four", "five", "six", "seven", "eight","nine", "ten", 
"eleven", "twelve", "thirteen", "fourteen", "fifteen","sixteen", "seventeen", "eighteen", 
"nineteen","twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", 
"hundred","thousand", "um",'know','see','oh', 'dot', 'com', 'customer', 'service', 'little', 
'bit','anything','else','take','look','first','ma'}



#-------------------------------------------------------------------------------------------------------------
#SUBSETTING DATA
#-------------------------------------------------------------------------------------------------------------
df_trans_req = df_trans[['a','b']]

#-------------------------------------------------------------------------------------------------------------
#PRE-PROCESSING THE DATA
#-------------------------------------------------------------------------------------------------------------
print("Performing initial data cleaning")

def corp_santz(corp):                    #this func replaces all non alphabet characters with extra space except single quote
    corp  = corp.lower() 
    text = re.sub("[^a-z']",' ',corp)
    text = re.sub("(can't)","can not",text)
    text = re.sub("(n't)"," not",text)
    text = re.sub("('s)","",text)
    return text

def corp_santz_case(corp):                    #this func replaces all non alphabet characters with extra space except single quote
    text = re.sub("[^A-Za-z']",' ',corp)
    text = re.sub("(can't)","can not",text)
    text = re.sub("(n't)"," not",text)
    text = re.sub("('s)","",text)
    return text

def missing_value(data):
    data = data.dropna(how='any') # Removed NaNs 
    data = data.reset_index(drop=True) #Reset Index of dataframe after dropping values
    return data

def tk_bigram(tk_unigram): #This func takes unigrams as input and returns Bigrams list
    tk_bgm  = nltk.bigrams(tk_unigram)
    bigram_list = []
    for i in tk_bgm:
        bigram_list.append(i)
    return bigram_list

def tk_trigram(tk_unigram):  #This func takes unigrams as input and returns Trigrams list
    tk_tgm = nltk.trigrams(tk_unigram)
    trigram_list = []
    for i in tk_tgm:
        trigram_list.append(i)
    return trigram_list



def word_tk(corp,gram = 3):   #This func takes raw text & type of tokens to return '1' - Unigrams/'2' - Bigrams/'3' - Trigrams list    
    stop_wd    = set(stopwords.words('english'))     #convert list of stopwords from nltk.corpus package in to 'set' to improve iteration speed
    exclude_stpwd = {"but","after","not","until","why","while","if"}
    stop_wd    = add_stoplist + stop_wd
    stop_wd    = stop_wd - exclude_stpwd
    tk         = nltk.word_tokenize(corp_santz(corp),language = 'english') #creates word tokens - unigrams 
    key_tokens = [word for word in tk if (word not in stop_wd and len(word) > 1)] # removes stop words and tokens with word_length = 1
    if gram == 2:                          #appends Bigrams to the list of unigrams
        key_tokens = key_tokens.append(tk_bigram(key_tokens))
    if gram == 3:                          #appends Bigrams and Trigrams to the list of unigrams 
        key_tokens = key_tokens + tk_bigram(key_tokens) + tk_trigram(key_tokens)
    return key_tokens


df = sanitize(df_trans_req,'name')
df = missing_value(df) #coulmn 'name' has 30172 and 'gender' has 30227, Name has NaNs

corp     = df.to_string(index=False,columns=["Review"],sparsify = False,
                                        index_names = False,header = False) #Converts all respondents reviews in to a string


#=== Unigrams ===#
tokens       = word_tk(corp,1)                #word tokens 
freq         = nltk.FreqDist(tokens)
freq_df      = pd.DataFrame(freq.items(),columns = ['Tokens','Frequency'])
freq_df.sort_values(['Frequency'],ascending = False).to_csv("Tokens.csv",encoding = 'utf-8')


#-------------------------------------------------------------------------------------------------------------
#RUN MODELS
#-------------------------------------------------------------------------------------------------------------
t0 = time()
print("Running unsupervised models")

vectorizer = CountVectorizer(tokenizer = word_tk,min_df=1
                                ,lowercase=True,binary = True,vocabulary = list(final_tk_lst.Tokens))

data_vectorized = vectorizer.fit_transform(corp)
    

# Define Search Param
search_params = {'n_components': [10], 'learning_decay': [.5]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vectorized)


# Best Model
best_lda_model = model.best_estimator_

# Show top n keywords for each topic
def show_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=20)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords
									 
# Create Document - Topic Matrix
lda_output = best_lda_model.transform(data_vectorized)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

# index names
docnames = df_merged["InteractionID"]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic
#-------------------------------------------------------------------------------------------------------------
#COMBINE RESULTS AND OUTPUT TABLE
#-------------------------------------------------------------------------------------------------------------
model_result = df_topic_keywords

import datetime
now = datetime.datetime.now()
curr_time = now.strftime("%Y_%m_%d_%H_%M")

model_result.to_csv(output_data_path +"/Topic_words_results_"+curr_time +".csv")
df_document_topic.to_csv(output_data_path +"/Doc_Topic_results_"+curr_time +".csv")
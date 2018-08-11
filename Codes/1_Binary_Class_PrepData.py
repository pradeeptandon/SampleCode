# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 14:36:08 2018

@author: bm30785
"""
#-------------------------------------------------------------------------------------------------------------
#DEFINE WORKING FOLDER and IMPORT PACKAGES
#-------------------------------------------------------------------------------------------------------------
import os
os.chdir("/home/bm30785/Codes/Transfer")

import nltk
import pickle
import pandas as pd
import numpy as np
from time import time
import re

from modules.dataimport import DataImport
#from modules import transcript_clean as trans_clean
from modules import text_normalization as norm
from modules import supplementary_functions as sf

#Import custom dictionary
from modules.lib.custom_ontology import __dict_list__

# Program start time
t_int = time()

#-------------------------------------------------------------------------------------------------------------
#SET INPUT and OUTPUT FOLDER PATHS
# Change the folder paths here for transcripts folder, metadata and output folder
#-------------------------------------------------------------------------------------------------------------
trans_data_path = "/home/bm30785/Data/Transcripts/Nexidia_DEU_Export_UAT/TranscriptData"
trans_data_path2 = "/home/bm30785/Data/Transcripts/Nexidia_DEU_Export_UAT/TranscriptData/Others"
interact_data_path="/home/bm30785/Data/Metadata"
output_data_path = "/home/bm30785/Data/PK_Data"

#-------------------------------------------------------------------------------------------------------------
#SUPPLEMENTARY FUNCTIONS
#-------------------------------------------------------------------------------------------------------------
## Function to merge and collapse transcripts by Interaction
def merge_transcript(df, df_interact):
    _df =  pd.DataFrame(df.groupby('InteractionID')['Script'].agg(lambda col: sum(col, [])))
    _df["InteractionID"] = _df.index
    _df.columns = [u'Script',u'InteractionID']
    
    ##Merging transcripts with interaction
    _df_merged = pd.merge(_df, df_interact, left_on = "InteractionID", right_on = "InteractionId" , how = 'left')
    _df_merged = _df_merged[["InteractionID","Complaints_ID","Script", u'Portfolio', u'BusName', u'FuncName', u'SubFunction']] 
    return _df_merged

#-------------------------------------------------------------------------------------------------------------
#IMPORT DATASETS
#-------------------------------------------------------------------------------------------------------------
# Importing (1) Transcription data from zip files and (2) Interaction Metadata
print("Importing transcript data files")
df_trans_raw = DataImport(trans_data_path, ["zip"]).read_file() #Currently limited to 20 files for testing purposes
df_trans_raw2 = DataImport(trans_data_path2, ["zip"]).read_file() #Remaining files
df_trans_raw = df_trans_raw.append(df_trans_raw2)

df_trans_raw.columns = [u"InteractionID",u'Speaker',u'OffsetStart',u'OffsetEnd',u'Script',u'Flag',u'DocID',u'DocName'] #Rename columns
print(df_trans_raw.head())
df_trans = df_trans_raw.copy(deep=True)

#Custom Dictionary
add_stoplist = {}
prefix_list = {"Agent: ", "Customer: "}

# Getting a unique list of Interaction IDs with transcript files
df_zip_interactID = df_trans_raw[["DocName", "InteractionID"]].drop_duplicates()
print("Imported "+ str(len(set(df_zip_interactID.InteractionID))) + " Interactions from " + str(len(set(df_zip_interactID.DocName))) + " files")
df_zip_interactID.head()

# Importing interaction metadata file
print("Importing Interaction metadata")
df_interact = DataImport(interact_data_path, ["xlsx"]).read_file()

#-------------------------------------------------------------------------------------------------------------
#SUBSETTING DATA
#-------------------------------------------------------------------------------------------------------------
# Including portfolio field to limit to Branded Cards
df_interact = df_interact[["InteractionId","Complaints_ID", u'Portfolio', u'BusName', u'FuncName', u'SubFunction']]

# Limit to Interactions in the transcript dataframe sample
ID_list = set(df_trans["InteractionID"])
df_interact.columns
df_interact2 = df_interact[df_interact.InteractionId.isin(ID_list)]
df_interact2["InteractionId"].head()
print(df_interact2.columns)

#Limit to Branded Cards
df_interact2 = df_interact2[df_interact2[u'Portfolio'].isin(['CitiBrands','ThankYou','Costco'])]
print("Distribution of complaints vs non-complaints, Limited to Branded Cards")
print(df_interact2.groupby(['Complaints_ID']).size())

#Limit to Non-Collection Calls
df_interact2 = df_interact2[df_interact2[u'SubFunction'].isin(['Rewards','Costco','Co-Brand','TY Customer Service', 'Blend - Costco Cards Coll', 'Unverified Cell - Costco Cards Coll'])]
print("Distribution of complaints vs non-complaints, Excluding collection calls")
print(df_interact2.groupby(['Complaints_ID']).size())

#-------------------------------------------------------------------------------------------------------------
#SPLITTING DATA BETWEEN COMPLAINTS VS NON-COMPLAINTS
#-------------------------------------------------------------------------------------------------------------
#Count of complaints vs non-complaints
df_interact2["Complaint_Flag"] = df_interact2.Complaints_ID.apply(lambda x: 1 if x == "Complaint" else 0)

#Number of non-complaints
nc_count = len(df_interact2)-df_interact2["Complaint_Flag"].sum()
print("Number of non-complaints")
print(nc_count)

#Subset equal number of sample for complaints
df_complaint_corpus=df_interact2[df_interact2.Complaint_Flag == 1 ]

df_complaint_corpus=df_complaint_corpus.sample(nc_count , random_state = 52)
print("Number of complaints in the sample")
print(len(df_complaint_corpus))

df_noncomplaint_corpus=df_interact2[df_interact2.Complaint_Flag == 0]
print("Number of non-complaints in the sample")
print(len(df_noncomplaint_corpus))

df_combined = df_complaint_corpus.append(df_noncomplaint_corpus)
ID_list = set(df_combined["InteractionId"])

#Filter transcript data
df_trans = df_trans[df_trans["InteractionID"].isin(ID_list)]

#-------------------------------------------------------------------------------------------------------------
#PRE-PROCESSING THE DATA
#-------------------------------------------------------------------------------------------------------------
print("Performing initial data cleaning")
##Check for float and convert to string
initial_sent = len(df_trans)
df_trans["sent_len"] = df_trans["Script"].apply(lambda x: len(str(x).split()))

##Remove <UNK> and "..." from the transcript
df_trans.Script = df_trans.Script.apply(lambda x: norm.remove_tags(str(x)))
df_trans.Script = df_trans.Script.apply(lambda x: x.replace("..."," "))
df_trans.Script = df_trans.Script.apply(lambda x: re.sub( '\s+', ' ', x).strip())
print(df_trans.head())

##Limit to sentences of more than 10 words
#word_limit = 4
#df_trans = df_trans[df_trans.sent_len > word_limit]
#final_sent = len(df_trans)

#print("Dropped %d sentences less than or equal to %d words" % (initial_sent - final_sent, word_limit))
df_trans["Script"] = df_trans["Script"].apply(lambda x: [str(x)] if sf.is_number(x) else [x])
#print("Total number of sentences greater than %d words: %d"%(word_limit, initial_sent))

#Limit to agent interactions only
df_trans = df_trans.copy()
#df_trans = df_trans[df_trans["Speaker"] == "Customer"]

#Sort transcript data
df_trans = df_trans.sort_values(by=['InteractionID','Speaker','OffsetStart'], ascending=[1,1,1])
print(df_trans.head())

#Merge transcript
#df_merged = merge_transcript(df_trans, df_interact)
df_merged = merge_transcript(df_trans, df_interact2)
#df_merged_customer = merge_transcript(df_trans_customer, df_interact)

print("Performing preprocessing")
##Remove_tags, remove_characters_before_tokenization, replace_words_using_dict, tokenize_text,remove_characters_after_tokenization
t0 = time()
df_merged ["Clean_Script"] = df_merged ["Script"].apply(lambda x: str(' '.join(x)))
df_merged = df_merged.copy()


df_merged ["Clean_Script"] = df_merged ["Clean_Script"].apply(lambda x: norm.preprocess(x, prefix = prefix_list, dict_list=__dict_list__, std_type="l"))  
df_merged ["Clean_Script"] = df_merged ["Clean_Script"].apply(lambda x: nltk.word_tokenize(x))

print("Preprocessing done in %0.3fs." % (time() - t0))
print(df_merged.head())

print("Removing Stopwords")
t0 = time()
add_stoplist={"ok", "thank","much", "gon", "wan", "go", "yes","madam", "like","na", "ahead", 
"wan", "let", "u", "would","one","moment", "make", "sure", "dollar", "name","please",
"zero", "one", "two", "three", "four", "five", "six", "seven", "eight","nine", "ten", 
"eleven", "twelve", "thirteen", "fourteen", "fifteen","sixteen", "seventeen", "eighteen", 
"nineteen","twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", 
"hundred","thousand", "um",'know','see','oh', 'dot', 'com', 'customer', 'service', 'little', 
'bit','anything','else','take','look','first','ma'}

df_merged ["Clean_Script_nostop"] = df_merged ["Clean_Script"].apply(lambda x: sf.remove_stopwords_list(x, add_stoplist))
print("Removing Stopwords done in %0.3fs." % (time() - t0))

#-------------------------------------------------------------------------------------------------------------
#SAVE THE DATAFRAME
#-------------------------------------------------------------------------------------------------------------
print("Saving dataframe")
t0 = time()
df_merged.to_pickle(output_data_path + "/merged_binclass_data_full.pk")

print("Done in %0.3fs." % (time() - t0))






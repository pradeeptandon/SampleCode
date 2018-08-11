# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:17:34 2018

@author: bm30785
"""
"""
--------------------------------------------------------------------------------------------
Supplementary functions
--------------------------------------------------------------------------------------------
"""

from time import time
from nltk.tokenize import word_tokenize

#Check for float and convert to string
def is_number(n):
    try:
        complex(n)
    except ValueError:
        return False

    return True

from nltk.corpus import stopwords
def remove_stopwords_list(wd_list, add_stop=()):
    output = [];
    stop = set(stopwords.words('english'))  # List of English common stopwords
    # Removes English stopwords, any additional stopwords provided and any term less than 3 letters
    for sentence in wd_list:
        text = [w for w in word_tokenize(sentence) if (not w.lower() in (set(stop) | set(add_stop)))]
        sent = " ".join(text)
        output.append(sent)
    return output


def text2int(textnum, numwords={}):
    if not numwords:
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                raise Exception("Illegal word: " + word)

            scale, increment = numwords[word]

        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current

#def change2num(token_list):
#    for word in token_list:
#        text = [w for w in token_list if (not w.lower() in (set(stop) | set(add_stop)))]
# 
#    


#Testing transcript cleaning 

#df_test = df_trans[df_trans["InteractionID"] == 68267]   
#df_out = clean_transcript(df_test, threshold = 0)
#df_out.head()


"""
-------------------------------------------------------------------------------------------
0. Data Exploration Testing
--------------------------------------------------------------------------------------------
"""
         
#Review scripts
#ID_list = [69153,70418,72984,75672,82964,94661,97521,103408,105454,106926,113310,117011,117111,117196,119297,120984,127355,129276,130324,131814]
#
#for select_ID in ID_list:
#    print(df_trans2.Clean_Script[select_ID])
#
#"""


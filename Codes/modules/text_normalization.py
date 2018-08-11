# -*- coding: utf-8 -*-
#=========================================================================
"""
This module is used for text normalization. It includes
1. Removing prefixes, punctuations, number etc.
2. Expanding contractions
3. Case conversions
4. Tokenization
5. Removing stopwords
6. Correcting spellings
7. Replacing words from custom dictionary
8. Stemming/Lemmatization
"""
#=========================================================================
import nltk
import re
import string
from time import time
#import html
import os
from nltk.tokenize import word_tokenize
#from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#from html.parser import HTMLParser
os.chdir('C:/Users/vt96899/Desktop/Complaint Management/KT/Codes')
from contractions import CONTRACTION_MAP

def remove_prefix(text, prefix_list, repl_val=""):
    """
    Remove the prefix from the string
    """
    if prefix_list != ():
        for term in prefix_list:
            text = re.sub("^s*"+term,"",text)
    return text

def remove_tags(text, repl_val=""):
    """
    Remove the tags from the string
    """
    tag_1 = re.compile(r'<[^>]+>')
    tag_2 = re.compile(r'\[[^\]]+\]')
    return tag_1.sub(repl_val, tag_2.sub(repl_val, text))
    
def remove_punctuation(text):   ## input is sentence
    """
    Remove punctuation characters from the string
    """
#    table = str.maketrans({key: None for key in string.punctuation})
#    return text.translate(table) ## output is sentence without punctuation
    return re.sub(r'[^\w\s]','',text)
    
    
def remove_int(text):           ###input is sentence
    """
    Remove any standalone numbers from the string
    """
    no_integer = [word for word in tokenize_text(text) if not (word.isdigit() or word[0]=='-' and word[1:].isdigit())]
    
    return " ".join(no_integer)

def remove_numbers(text):
    """
    Remove any numbers and keep only text
    """
    text = re.sub("[^a-zA-Z\s]", "", text)  # Keep letters only 
    return text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    """
    Expand contractions in the string using pre-defined library
    """
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_sentence = contractions_pattern.sub(expand_match, text)
    return expanded_sentence
           
def tokenize_text(text):
    """
    Tokenize text using NLTK
    """
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences] 
    word_tokens = [val for sublist in word_tokens for val in sublist]#Flatten List
    return word_tokens
 
def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens
    
def remove_characters_before_tokenization(text, keep_apostrophes=False):
    text = text.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~|,]'
        filtered_sentence = re.sub(PATTERN, r'', text)
    else:
        PATTERN = r'[^a-zA-Z0-9 ]'
        filtered_sentence = re.sub(PATTERN, r'', text)
    return filtered_sentence

def lowercase(text):       ##input is list of words -- returns list of words in lowercase
    return text.lower()

def remove_stopwords(text, add_stop=()):
    stop = set(stopwords.words('english'))  # List of English common stopwords
    # Removes English stopwords, any additional stopwords provided and any term less than 3 letters
    text = [w for w in word_tokenize(text) if (not w in (set(stop) | set(add_stop)))]
    return ' '.join(text)

#def remove_repeated_characters(tokens):
#    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
#    match_substitution = r'\1\2\3'
#    def replace(old_word):
#        if wordnet.synsets(old_word):
#            return old_word
#        new_word = repeat_pattern.sub(match_substitution, old_word)
#        return replace(new_word) if new_word != old_word else new_word
#            
#    correct_tokens = [replace(word) for word in tokens]
#    return correct_tokens

def replace_words_using_dict(text,dict_list):
    for key in dict_list: 
        text = text.replace(key, dict_list[key])
    return text

def standardize(text, stem_type=None):
    tokens = nltk.word_tokenize(text)
    if stem_type == "s":
        stemmer = PorterStemmer()
        out =  [stemmer.stem(w) for w in tokens]
    elif stem_type == "l":
        stemmer = WordNetLemmatizer()
        out =  [stemmer.lemmatize(w) for w in tokens]
    else:
        out =  tokens
    return " ".join(out)

#def unescape_html(text):
#    return html.unescape(text)

def parse_string(text):
    text = re.sub('\n', ' ', text)
    if isinstance(text, str):
        text = text
    elif isinstance(text, unicode): 
        str(text, 'utf-8')
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
    else:
        raise ValueError('Document is not string or unicode!')
    text = str(text).strip()
#    text = unescape_html(text)
#    sentences = nltk.sent_tokenize(text)
#    sentences = [sentence.strip() for sentence in sentences]
    return text

#Preprocess the dataframe
def preprocess(text, prefix=(), dict_list=(), add_stoplist=(), std_type = "l"):
    text = remove_tags(text)
#    text = parse_string(text)
#    text = remove_prefix(text ,prefix)
    text = expand_contractions(text)
    text = remove_characters_before_tokenization(text, keep_apostrophes=True)
    text = lowercase(text)
    text = replace_words_using_dict(text,dict_list)  
#    text = remove_stopwords(text, add_stoplist)  
    text = remove_int(text)
    text = standardize(text, stem_type=std_type)

#    text = remove_numbers(text)
    text = tokenize_text(text)
    text = remove_characters_after_tokenization(text)

    return " ".join(text)



    
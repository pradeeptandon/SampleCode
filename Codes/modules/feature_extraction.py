# -*- coding: utf-8 -*-
#=========================================================================
"""
This module is used for feature extraction as inputs to classification models.
The feature extraction techniques that are applied are:
1. Bag of Words
2. TF-IDF
3. Word2Vec Models
    3a. Average Word Vectors
    3b. TF-IDF Weighted Average Word Vectors 
"""
#=========================================================================
import pandas as pd
import gensim
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from scipy.sparse.linalg import svds

# Bag of Words Model
def bow_extractor(corpus, ngram_range=(1,1), max_features=None):
    vectorizer = CountVectorizer(min_df=1, max_features = None, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

def display_features(features, feature_names):
    df = pd.DataFrame(data=features,columns=feature_names)
    print (df)
    
# TF-IDF Model
def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',smooth_idf=True,use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix

def tfidf_extractor(corpus, ngram_range=(1,1), max_features=None):
    vectorizer = TfidfVectorizer(min_df=1,
    norm='l2',
    smooth_idf=True,
    use_idf=True,
    ngram_range=ngram_range,
    max_features=None)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# Advanced Word Vectorization Models
def build_word2vec(TOKENIZED_CORPUS, size=10, window=10,min_count=2, sample=1e-3):
    """
    # Build the word2vec model on our training corpus. It creates a vector representation for each word in the vocabulary.
    """
    model = gensim.models.Word2Vec(TOKENIZED_CORPUS, size, window,min_count, sample)
    return model

# define function to average word vectors for a text document
def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
#    print(words)
    for word in words:
        if word in vocabulary:
            if len(model[word]) == num_features:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model[word])
                if nwords:
                    feature_vector = np.divide(feature_vector, nwords)
#                    print(word, len(feature_vector))
    return feature_vector

# define function to average word vectors for a text document
#def average_word_vectors(words, model, vocabulary, num_features):
#    doc = [word for word in words if word in vocabulary]
#    return np.mean(model[doc], axis = 0)

# generalize above function for a corpus of documents
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary,num_features) for tokenized_sentence in corpus]
    return np.array(features)

# TF-IDF Weighted Averaged Word Vectors
def tfidf_wtd_avg_word_vectors(words, tfidf_vector, tfidf_vocabulary, model,num_features):
    """
    Define function to compute tfidf weighted averaged word vector for a document
    """
    word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)] if tfidf_vocabulary.get(word) else 0 for word in words]
    word_tfidf_map = {word:tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}
    feature_vector = np.zeros((num_features,),dtype="float64")
    vocabulary = set(model.wv.index2word)
    wts = 0.
    for word in words:
        if word in vocabulary:
            word_vector = model[word]
            weighted_word_vector = word_tfidf_map[word] * word_vector
            wts = wts + word_tfidf_map[word]
            feature_vector = np.add(feature_vector, weighted_word_vector)
    if wts:
        feature_vector = np.divide(feature_vector, wts)
    return feature_vector

# generalize above function for a corpus of documents
def tfidf_weighted_averaged_word_vectorizer(corpus, tfidf_vectors,tfidf_vocabulary, model, num_features):
    docs_tfidfs = [(doc, doc_tfidf) for doc, doc_tfidf in zip(corpus, tfidf_vectors)]
    features = [tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary,model, num_features) for tokenized_sentence, tfidf in docs_tfidfs]
    return np.array(features)

# Generate feature matrix
def build_feature_matrix(documents, feature_type='frequency'):
    feature_type = feature_type.lower().strip()
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=1,ngram_range=(1, 1))
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=1,
        ngram_range=(1, 1))
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=1,
        ngram_range=(1, 1))
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")
    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    return vectorizer, feature_matrix

def low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt
    
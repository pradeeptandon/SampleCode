# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:20:43 2018

@author: bm30785
"""
import pickle
import time
import numpy as np
import nltk
import gensim
from pylab import *
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


from modules import feature_extraction as fe

def prepare_datasets(corpus, labels, proportion = 0.3):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels, test_size=proportion, random_state=42)
    print("  %d training examples (%d Yes)" % (len(train_Y), sum(train_Y)))
    print("  %d test examples (%d Yes)" % (len(test_Y), sum(test_Y)))
    return train_X, test_X, train_Y, test_Y

def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels
    
def get_metrics(true_labels, predicted_labels):
    
    print ('Accuracy: %s' % str(np.round(metrics.accuracy_score(true_labels,predicted_labels),2)))
    print ('Precision: %s' % str(np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 2)))
    print ('Recall: %s' %str(np.round( metrics.recall_score(true_labels, predicted_labels, average='weighted'), 2)))
    print ('F1 Score: %s' %str(np.round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 2)))
                        
def train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels):
    
    classifier.fit(train_features, train_labels) # build model      
    predictions = classifier.predict(test_features)  # predict using model
    # evaluate model prediction performance   
    
    print ("------------------------------------------------------------------------------------------------")
    print(get_metrics(true_labels=test_labels,predicted_labels=predictions))
    print(metrics.classification_report(test_labels, predictions))
    print
    print ("Confusion matrix")
    print(metrics.confusion_matrix(test_labels, predictions))
    print ("------------------------------------------------------------------------------------------------")

#    return predictions    

def LSA_vectorizer(X_train, X_test, max_features=10000, red_features = 100, ngram_range=(1,1)):

    # Tfidf vectorizer:
    #   - Strips out “stop words”
    #   - Filters out terms that occur in more than half of the docs (max_df=0.5)
    #   - Filters out terms that occur in only one document (min_df=2).
    #   - Selects the 10,000 most frequently occuring words in the corpus.
    #   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of 
    #     document length on the tf-idf values. 
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,  min_df=3, stop_words='english', use_idf=True, ngram_range=ngram_range)
    
    # Build the tfidf vectorizer from the training data ("fit"), and apply it ("transform").
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print("  Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])
    
    
    print("\nPerforming dimensionality reduction using LSA")
    t0 = time.time()
    
    # Project the tfidf vectors onto the first N principal components.
    # Though this is significantly fewer features than the original tfidf vector,
    # they are stronger features, and the accuracy is higher.
    svd = TruncatedSVD(red_features)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    
    # Run SVD on the training data, then project the training data.
    X_train_lsa = lsa.fit_transform(X_train_tfidf)
    
    print("Done in %.3fsec" % (time.time() - t0))
    
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

    # Now apply the transformations to the test data as well.
    X_test_tfidf = vectorizer.transform(X_test)
    X_test_lsa = lsa.transform(X_test_tfidf)
    
    return X_train_lsa, X_test_lsa


def inspect_LSA(X_train_raw, red_features=100, firstN = 10, ngram_range=(1,1)):
    ###############################################################################
    #  Use LSA to vectorize the articles.
    ###############################################################################
    
    # Tfidf vectorizer:
    #   - Strips out “stop words”
    #   - Filters out terms that occur in more than half of the docs (max_df=0.5)
    #   - Filters out terms that occur in only one document (min_df=2).
    #   - Selects the 10,000 most frequently occuring words in the corpus.
    #   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of 
    #     document length on the tf-idf values. 
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True, ngram_range=ngram_range)
    
    # Build the tfidf vectorizer from the training data ("fit"), and apply it 
    # ("transform").
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    
    print("Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])
    
    # Get the words that correspond to each of the features.
    feat_names = vectorizer.get_feature_names()
    
#    # Print ten random terms from the vocabulary
#    print("Some random words in the vocabulary:")
#    for i in range(0, 10):
#        featNum = random.randint(0, len(feat_names))
#        print("  %s" % feat_names[featNum])
#        
    print("\nPerforming dimensionality reduction using LSA")
    #t0 = time.time()
    
    # Project the tfidf vectors onto the first N principal components.
    # Though this is significantly fewer features than the original tfidf vector,
    # they are stronger features, and the accuracy is higher.
    svd = TruncatedSVD(red_features)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    
    # Run SVD on the training data, then project the training data.
    X_train_lsa = lsa.fit_transform(X_train_tfidf)
    
    # The SVD matrix will have one row per component, and one column per feature
    # of the original data.
    
    #for compNum in range(0, 100, 10):
    for compNum in range(0, firstN):  
        comp = svd.components_[compNum]       
        # Sort the weights in the first component, and get the indeces
        indeces = np.argsort(comp).tolist()       
        # Reverse the indeces, so we have the largest weights first.
        indeces.reverse()
        
        # Grab the top 10 terms which have the highest weight in this component.        
        terms = [feat_names[weightIndex] for weightIndex in indeces[0:10]]    
        weights = [comp[weightIndex] for weightIndex in indeces[0:10]]    
       
        # Display these terms and their weights as a horizontal bar graph.    
        # The horizontal bar graph displays the first item on the bottom; reverse
        # the order of the terms so the biggest one is on top.
        terms.reverse()
        weights.reverse()
        positions = np.arange(10) + .5    # the bar centers on the y axis
        
        figure(compNum)
        barh(positions, weights, align='center')
        yticks(positions, terms)
        xlabel('Weight')
        title('Strongest terms for component %d' % (compNum))
        grid(True)
        show()
    return X_train_lsa


def skclassify_models(corpus, labels,  proportion = 0.3, model = None, num_features = 100, ngram_range = (1,1), lsa_features = 100):  
    # Prepare dataset
    train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus, labels, proportion)
                                                                        
    # Bag of words features
    bow_vectorizer, bow_train_features = fe.bow_extractor(train_corpus, ngram_range, max_features = num_features)  
    bow_test_features = bow_vectorizer.transform(test_corpus) 
    
    # tfidf features
    tfidf_vectorizer, tfidf_train_features = fe.tfidf_extractor(train_corpus, ngram_range,max_features = num_features)  
    tfidf_test_features = tfidf_vectorizer.transform(test_corpus)    
    
    # Tokenize documents
#    tokenized_train = [nltk.word_tokenize(text) for text in train_corpus]
#    tokenized_test = [nltk.word_tokenize(text) for text in test_corpus]  
#
#    #build word2vec model  
#    if model == None:                 
#        model = gensim.models.Word2Vec(tokenized_train, size=num_features, window=10, min_count=30, sample=1e-3)    
#    else:
#        model = model              
#                       
#    # Averaged word vector features
#    avg_wv_train_features = fe.averaged_word_vectorizer(corpus = tokenized_train, model = model, num_features = num_features)                   
#    avg_wv_test_features = fe.averaged_word_vectorizer(corpus = tokenized_test, model = model, num_features = num_features)                                                 
#                       
#    ## tfidf weighted averaged word vector features
#    vocab = tfidf_vectorizer.vocabulary_
#    tfidf_wv_train_features = fe.tfidf_weighted_averaged_word_vectorizer(corpus = tokenized_train, tfidf_vectors = tfidf_train_features, tfidf_vocabulary=vocab, model=model, num_features=num_features)
#    tfidf_wv_test_features = fe.tfidf_weighted_averaged_word_vectorizer(corpus = tokenized_test, tfidf_vectors = tfidf_test_features, tfidf_vocabulary=vocab, model=model, num_features=num_features)
    
    ## LSA reducted features
    
    X_train_LSA, X_test_LSA = LSA_vectorizer(train_corpus, test_corpus, red_features = lsa_features, ngram_range = ngram_range)
    
    #Define classifer model
    lgr = LogisticRegression()
    dct = DecisionTreeClassifier()
    mnb = MultinomialNB()
    svm = SGDClassifier(loss='hinge', n_iter=100)
    #mlp = MLPClassifier(solver='lbfgs', alpha = 1e-5, hidden_layer_sizes = (1000,), random_state=1)
    #mlp2 = MLPClassifier(solver='lbfgs', alpha = 1e-5, hidden_layer_sizes = (250 , 250, 2), random_state=1)
    rdf = RandomForestClassifier(n_jobs=2, random_state = 0)
    eclf = VotingClassifier(estimators=[('lgr',lgr), ('dct',dct), ('rdf',rdf)], voting = 'soft', weights = [1,1,1])
    eclf2 = VotingClassifier(estimators=[('lgr',lgr), ('svm',svm)], voting = 'hard', weights = [1000, 20])
    
       
    def sk_model_run(classifier, train_features, train_labels, test_features, test_labels, labels):
        print(labels)
        train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels)
    
    # With bag of words features
    sk_model_run(lgr, bow_train_features, train_labels, bow_test_features, test_labels,"Using Logistic Regression with bag of words features")
    sk_model_run(dct, bow_train_features, train_labels, bow_test_features, test_labels,"Using Decision Tree with bag of words features")
    sk_model_run(mnb, bow_train_features, train_labels, bow_test_features, test_labels,"Using Multinomial Naive Bayes with bag of words features")
    sk_model_run(svm, bow_train_features, train_labels, bow_test_features, test_labels,"Using Support Vector Machine with bag of words features")
    #sk_model_run(mlp, bow_train_features, train_labels, bow_test_features, test_labels,"Using Neural Network (1 Hidden Layer, 1k units) with bag of words features")
    #sk_model_run(mlp2, bow_train_features, train_labels, bow_test_features, test_labels,"Using Neural Network (2 Hidden Layers, .75k units) with bag of words features")
    sk_model_run(rdf, bow_train_features, train_labels, bow_test_features, test_labels,"Using Random Forest with bag of words features")
    sk_model_run(eclf, bow_train_features, train_labels, bow_test_features, test_labels,"Using Ensembled Models (Soft) with bag of words features")
    sk_model_run(eclf2, bow_train_features, train_labels, bow_test_features, test_labels,"Using Ensembled Models (Hard) with bag of words features")
    
    input("Press Enter to Continue...")

    # With TF-IDF features
    sk_model_run(lgr, tfidf_train_features, train_labels, tfidf_test_features, test_labels,"Using Logistic Regression with TF-IDF features")
    sk_model_run(dct, tfidf_train_features, train_labels, tfidf_test_features, test_labels,"Using Decision Tree with TF-IDF features")
    sk_model_run(mnb, tfidf_train_features, train_labels, tfidf_test_features, test_labels,"Using Multinomial Naive Bayes with TF-IDF features")
    sk_model_run(svm, tfidf_train_features, train_labels, tfidf_test_features, test_labels,"Using Support Vector Machine with TF-IDF features")
    #sk_model_run(mlp, tfidf_train_features, train_labels, tfidf_test_features, test_labels,"Using Neural Network (1 Hidden Layer, 1k units) with bag of words features")
    #sk_model_run(mlp2, tfidf_train_features, train_labels, tfidf_test_features, test_labels,"Using Neural Network (2 Hidden Layers, .75k units) with bag of words features")
    sk_model_run(rdf, tfidf_train_features, train_labels, tfidf_test_features, test_labels,"Using Random Forest with TF-IDF features")
    sk_model_run(eclf, tfidf_train_features, train_labels, tfidf_test_features, test_labels,"Using Ensembled Models (Soft) with TF-IDF features")
    sk_model_run(eclf2, tfidf_train_features, train_labels, tfidf_test_features, test_labels,"Using Ensembled Models (Hard) with TF-IDF features")

    input("Press Enter to Continue...")
      
    # With averaged word vector features
#    sk_model_run(lgr, avg_wv_train_features, train_labels, avg_wv_test_features, test_labels,"Using Logistic Regression with average word vector features")
#    sk_model_run(dct, avg_wv_train_features, train_labels, avg_wv_test_features, test_labels,"Using Decision Tree with average word vector features")
##    sk_model_run(mnb, avg_wv_train_features, train_labels, avg_wv_test_features, test_labels,"Using Multinomial Naive Bayes with average word vector features")
#    sk_model_run(svm, avg_wv_train_features, train_labels, avg_wv_test_features, test_labels,"Using Support Vector Machine with average word vector features")
#   
#    # With tfidf weighted averaged word vector features
#    sk_model_run(lgr, tfidf_wv_train_features, train_labels, tfidf_wv_test_features, test_labels,"Using Logistic Regression with tfidf weighted averaged word vector features")
#    sk_model_run(dct, tfidf_wv_train_features, train_labels, tfidf_wv_test_features, test_labels,"Using Decision Tree with tfidf weighted averaged word vector features")
##    sk_model_run(mnb, tfidf_wv_train_features, train_labels, tfidf_wv_test_features, test_labels,"Using Multinomial Naive Bayes with average word vector features")
#    sk_model_run(svm, tfidf_wv_train_features, train_labels, tfidf_wv_test_features, test_labels,"Using Support Vector Machine with tfidf weighted averaged word vector features")
#    
    # With LSA reduced features
    sk_model_run(lgr, X_train_LSA, train_labels, X_test_LSA, test_labels,"Using Logistic Regression with LSA features")
    sk_model_run(dct, X_train_LSA, train_labels, X_test_LSA, test_labels,"Using Decision Tree with with LSA features")
#    sk_model_run(mnb, tfidf_wv_train_features, train_labels, tfidf_wv_test_features, test_labels,"Using Multinomial Naive Bayes with average word vector features")
    sk_model_run(svm, X_train_LSA, train_labels, X_test_LSA, test_labels,"Using Support Vector Machine with LSA features")
#   sk_model_run(mlp, X_train_LSA, train_labels, X_test_LSA, test_labels,"Using Neural Network with LSA features")
#   sk_model_run(mlp2, X_train_LSA, train_labels, X_test_LSA, test_labels,"Using Neural Network 2 with LSA features")
    sk_model_run(rdf, X_train_LSA, train_labels, X_test_LSA, test_labels,"Using Random Forest with LSA features")
    sk_model_run(eclf, X_train_LSA, train_labels, X_test_LSA, test_labels,"Using Ensembled Models (Soft) with LSA features")
    sk_model_run(eclf2, X_train_LSA, train_labels, X_test_LSA, test_labels,"Using Ensembled Models (Hard) with LSA features")

    input("Press Enter to Continue...")
  
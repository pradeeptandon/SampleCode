# -*- coding: utf-8 -*-
#=========================================================================
# -*- coding: utf-8 -*-
"""
Spyder Editor

author: bm30785

Used for key phrases analysis
"""
#=========================================================================
import itertools
import nltk
#from gensim import corpora, models
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist, bigrams, trigrams
from operator import itemgetter
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder, TrigramAssocMeasures
from nltk.metrics.association import QuadgramAssocMeasures
#from textblob import TextBlob

#FLATTEN LIST
def flatten_list(list_of_list):
    flattened=[val for sublist in list_of_list for val in sublist]
    return flattened

def flatten_corpus(corpus):
    return ' '.join([document.strip() 
                     for document in corpus])

#COLLOCATIONS
def compute_ngrams(sequence, n):
    return zip(*[sequence[index:] 
                 for index in range(n)])
    
def frequent_words(df_series,num=20,flatten=False):        
    if flatten:
        freqdist = FreqDist(flatten_list(df_series))
    else:
        freqdist = FreqDist(df_series)  
    return(freqdist.most_common(num))

def get_top_ngrams(corpus, ngram_val=1, limit=5):

    corpus = flatten_corpus(corpus)
    tokens = word_tokenize(corpus)

    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(), 
                              key=itemgetter(1), reverse=True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq) 
                     for text, freq in sorted_ngrams]

    return sorted_ngrams

def get_n_grams(df_series_tokens, ngram_val=2, limit = 20):    
    '''
    This part will find top 5 bigrams and add features to dataframe
    '''
    text=flatten_list(df_series_tokens)
    if ngram_val == 2:
        x=list(bigrams(text))
    elif ngram_val == 3:
        x=list(trigrams(text))
    else:
        x=list(bigrams(text))
    freq_bi = FreqDist(x)
    most_freq_bi=[]
    for t1,t2 in freq_bi.most_common(limit):
        most_freq_bi.append(t1)
    most_freq_bigrams=[]    
    for tup in most_freq_bi:
        most_freq_bigrams.append(' '.join(tup))        
    return most_freq_bigrams
 
def find_ngram_terms(df_series_tokens, search_term, num=20, n_gram=2):
    text=flatten_list(df_series_tokens)
    if n_gram == 2:
        x=list(bigrams(text))
    elif n_gram == 3:
        x=list(trigrams(text))
    else:
        x=list(bigrams(text))
    x = [dup for dup in x if search_term in dup]
    freq_d = FreqDist(x)
    most_freq_n=[]
    for t1,t2 in freq_d.most_common(num):
        most_freq_n.append(t1)
    most_freq_ns=[]    
    for tup in most_freq_n:
        most_freq_ns.append(' '.join(tup))        
    return most_freq_ns

#Find collocations    
def col_locations(text, limit = 10, window_size = 2):
    tokenized_text = word_tokenize(text)
    bigram_measures = BigramAssocMeasures()
    
    finder = BigramCollocationFinder.from_words(tokenized_text, window_size)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    
    sorted(scored, key=lambda s: s[1], reverse=True)       
    return scored[:limit]


#Find collocations with given terms
def get_collocations(term1, term2 = None, sentence = "", ngram=2, window_size=3):
    words = nltk.wordpunct_tokenize(sentence)
    if ngram == 2:
        finder = nltk.collocations.BigramCollocationFinder.from_words(words, window_size = window_size)
        measures = nltk.collocations.BigramAssocMeasures()

    if ngram == 3:
        finder = nltk.collocations.TrigramCollocationFinder.from_words(words, window_size = window_size)
        measures = nltk.collocations.TrigramAssocMeasures()
        
    if ngram == 4:
        finder = nltk.collocations.QuadgramCollocationFinder.from_words(words, window_size = window_size)
        measures = nltk.collocations.TrigramAssocMeasures()

    for i in finder.nbest(measures.raw_freq, 10000):
        if (term2 != None):
            if (term1 in i) & (term2 in i):
                print (i)
        else:
            if (term1 in i):
                print (i)
    #get_collocations(term1, term2, sentence, ngram=4, window_size=5)
    
#Find nouns in the collocations
def get_nouns_in_collocation(term1, term2 = None, sentence="", ngram=2, window_size=3):
    words = nltk.wordpunct_tokenize(sentence)
    if ngram == 2:
        finder = nltk.collocations.BigramCollocationFinder.from_words(words, window_size = window_size)
        measures = nltk.collocations.BigramAssocMeasures()

    if ngram == 3:
        finder = nltk.collocations.TrigramCollocationFinder.from_words(words, window_size = window_size)
        measures = nltk.collocations.TrigramAssocMeasures()
        
    if ngram == 4:
        finder = nltk.collocations.QuadgramCollocationFinder.from_words(words, window_size = window_size)
        measures = QuadgramAssocMeasures()
    output = []
    
    for i in finder.nbest(measures.likelihood_ratio, 10000):
#        print(i)
        if (term2 != None):
            if (term1 in i) & (term2 in i):
#                print (i)
                for j in range(0,len(i)):
                   if (i[j] != term1) & (i[j] != term2):
                        nouns = [word for word,pos in nltk.pos_tag([i[j]]) if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
                        downcased = [x.lower() for x in nouns]
                        if nouns != []:
                            print(i)
                            output.append(downcased)
        else:
            if (term1 in i):
#                print(i)
                for j in range(0,len(i)):
                    if i[j] != term1:
                        nouns = [word for word,pos in nltk.pos_tag([i[j]]) if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
                        downcased = [x.lower() for x in nouns]
                        if nouns != []:
                            print(i)
                            output.append(downcased)
    return (set([val for sublist in output for val in sublist]))
#a = get_nouns_in_collocation("cancel", term2="card", sentence=sentence2, ngram=3, window_size=4)

def collocation_bymeasure(corpus, ngram_val=2, by_measure="raw_freq", limit=10):
    """ 
    To find collocations using various measures like raw frequencies, pointwise mutual information, and so on.
    """
    
    if ngram_val!=3:
        finder = BigramCollocationFinder.from_documents([item.split() for item in corpus])
        measures = BigramAssocMeasures()
    else:
        finder = TrigramCollocationFinder.from_documents([item.split() for item in corpus])
        measures = TrigramAssocMeasures()
        
    if by_measure == "raw_freq":                                             
        return finder.nbest(measures.raw_freq, limit)
    elif by_measure =="pmi":
        return finder.nbest(measures.pmi, limit)
    elif by_measure =="likelihood":
        return finder.nbest(measures.likelihood_ratio, limit)
    else:
        print("Measure type is not correct")

# Weighted Tag–Based Phrase Extraction  
def get_chunks(sentences, grammar = r'NP: {<DT>? <JJ>* <NN.*>+}', stopword_list = "Default"):
    """
    1. Extract all noun phrases chunks using shallow parsing
       2. Compute TF-IDF weights for each chunk and return the top weighted phrases
       """
       
    all_chunks = []
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    if stopword_list == "Default":
        stopword_list = nltk.corpus.stopwords.words('english')
    for sentence in sentences:
        tagged_sents = nltk.pos_tag_sents([nltk.word_tokenize(sentence)])
        chunks = [chunker.parse(tagged_sent) for tagged_sent in tagged_sents]
        wtc_sents = [nltk.chunk.tree2conlltags(chunk) for chunk in chunks]    
        flattened_chunks = list(itertools.chain.from_iterable(wtc_sent for wtc_sent in wtc_sents))
        valid_chunks_tagged = [(status, [wtc for wtc in chunk]) 
                                for status, chunk 
                                in itertools.groupby(flattened_chunks, key =  
                                                     lambda x: x[2] != 'O')]        
        valid_chunks = [' '.join(word.lower() 
                                for word, tag, chunk 
                                in wtc_group 
                                    if word.lower() 
                                        not in stopword_list) 
                                    for status, wtc_group 
                                    in valid_chunks_tagged
                                        if status]                                            
        all_chunks.append(valid_chunks)  
    return all_chunks


#def get_np(sentences, phrase = "NP", source = "Textblob"):
#    if type(sentences) == str:
#        sentences = sent_tokenize(sentences)
#    if source =="Textblob":
#        if phrase =="NP":
#            a = [TextBlob(str(i).lower()).noun_phrases for i in sentences]
#            lst = [str(i) for i in a if i!=[]]
#            return lst
#    else:
#        print ("System is missing")

        
#def get_tfidf_weighted_keyphrases(sentences,grammar=r'NP: {<DT>? <JJ>* <NN.*>+}',top_n=10):
#    # get valid chunks
#    valid_chunks = get_chunks(sentences, grammar=grammar)
#    # build tf-idf based model
#    dictionary = corpora.Dictionary(valid_chunks)
#    corpus = [dictionary.doc2bow(chunk) for chunk in valid_chunks]
#    tfidf = models.TfidfModel(corpus)
#    corpus_tfidf = tfidf[corpus]
#    # get phrases and their tf-idf weights
#    weighted_phrases = {dictionary.get(id): round(value,3)
#                        for doc in corpus_tfidf
#                        for id, value in doc}
#    weighted_phrases = sorted(weighted_phrases.items(),key=itemgetter(1), reverse=True)
#    # return top weighted phrases
#    return weighted_phrases[:top_n]
#!/usr/bin/env python

import re
import csv
import pickle
import requests
import json
import simplejson
import hashlib
import logging
import itertools
import ast
import types

import redis
import redis_collections as rc
import redis_wrap as rw
import hiredis
import msgpack

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


import numpy as np
import pandas as pd

import nltk as nl
from nltk.corpus import stopwords
import nltk.tokenize as nt
from nltk.tokenize import RegexpTokenizer

from textblob import blob, Blobber, TextBlob, Sentence, Word, WordList, tokenizers, sentiments, taggers, parsers, classifiers, wordnet
from textblob_aptagger import PerceptronTagger

class redisDataValidation(object):
    
    def __init__(self, hostname, port_num=6379, DB=0):
        self.r          =   redis.StrictRedis(host=hostname, port=port_num, db=DB)
        self.allkeys    =   self.r.keys('*')
        self.data       =   pd.DataFrame(self.allkeys, index=self.allkeys)
    
    ## Data profiling:
    #   1. List all ngrams with frequency of occurrance
    #   2. List all ngrams and frquency by prefix
    #
    
    
    pass

class PandasNgram(object):
    '''
    Class provides the functionality of taking in a recordset of strings from redis or potentially some other store and tokenize then generate n-grams.
    The results are stored in a pandas dataset.
    '''
    def __init__(self, hostname, port_num=6379, DB=0, stop_words=None):
        '''
        1.  Set up a database connection for a PandasNgram object.
        2.  Feed in a list of stop words or use a default if none are supplied.
        3.  Setup stemming and lemmatizing options
        '''
        
        self.r          =   redis.StrictRedis(host=hostname, port=port_num, db=DB)
        self.allkeys    =   self.r.keys('*')[:5000]
        
        if stop_words:
            self.stopWords  =   stop_words
        else:
            self.stopWords  =   stopwords.words('english')
        
        self.tokenizer  =   RegexpTokenizer(pattern=r'\w+')
        self.stemmer    =   nl.PorterStemmer()
        self.lemmatize  =   nl.WordNetLemmatizer()
        
        self.data   =   pd.DataFrame(self.allkeys, index=self.allkeys)
        self.ngram_collection =   pd.DataFrame(self.allkeys, index=self.allkeys)
        
        self.data.columns = ['key_id']
        self.ngram_collection.columns = ['key_id']
        
        self.offset =   0
        
        #self.re_URL = re.compile("^\s*URL.*$", re.MULTILINE)
        url_pattern = "((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?"
        self.re_URL = re.compile(url_pattern)
        self.re_TAG = re.compile("(<[phl]>)", re.IGNORECASE)
        self.re_WS = re.compile("/[^\S\n]/")
        self.re_DIGIT = re.compile("\d")
        self.re_CTRL = re.compile("[\x00-\x11\x03-\x1F]+")
        self.re_HI = re.compile("[\x80-\xFF]+")
        self.re_NWC = re.compile("[!;<>?{}\/~`#=@#$%^&*()_+]") #-

    def removeNoneTypes(self, lst):
        return [i for i in lst if type(i) is not types.NoneType]
        
    def unroll(self, line):
        if line:
            return line[0].values()[0]
        else:
            return None
        
    def unrollValues(self, name):
        '''
        Unroll the result from a msgpack formatted dict entity and get stored values
        '''
        #data = self.data[name][0].values()[0][0]
        data = self.data[name].map(lambda x: x.values()[0][0])
        data = self.data[name][0].values()[0][0]
        data = data.values()[0]
        return data.values()

    def unrollDictValues(self, name):
        '''
        Unroll the result from a msgpack formatted dict entity and get stored values
        '''
        data = self.data[name].map(lambda x: x.values()[0])
        data = data.map(lambda x: x.pop() if x else 'NA')
        data = data.map(lambda x: x.values()[0] if isinstance(x, dict) else 'NA')
        return data

    def unrollKeys(self, name):
        '''
        Unroll the result from a msgpack formatted dict entity and get stored keys
        '''
        data = self.data[name][0].values()[0][0]
        data = data.values()[0]
        return data.keys()

    def unrollDictKeys(self, name):
        '''
        Unroll the result from a msgpack formatted dict entity and get stored values
        '''
        data = self.data[name].map(lambda x: x.values()[0])
        data = data.map(lambda x: x.pop() if x else 'NA')
        data = data.map(lambda x: x.keys()[0] if isinstance(x, dict) else 'NA')
        return data

    def getRedisData(self, name):
        '''
        Retrieve all data for supplied keys contained in the 'named' redis bucket
        Uses the key_id as keys
        '''
        if (name in []):
            pass
        elif (name in ['headline','experience', 'education', 'skills', 'summary']):
            return self.data.key_id.map(lambda key: msgpack.unpackb(self.r.hget(key, name)))
        else:
            return self.data.key_id.map(lambda key: self.r.hget(key, name))
    
    def cleanText(self, prefix=None):
        '''
        Cleans text data by:
        1.  force lowercase
        2.  remove non-ascii chars
        3.  standardize whitespace
        4.  remove digits
        5.  remove control characters
        6.  remove URL patterns
        '''
        try:
            self.data[prefix]   =   self.data[prefix].map(lambda x: x.lower().decode('unicode_escape'))
        except UnicodeDecodeError, e:
            print(e)
            self.data[prefix]   =   self.data[prefix].map(lambda x: x.lower())
        except Exception, e:
            print(e)
        finally:
            self.data[prefix]   =   self.data[prefix].map(lambda x: x.lower())

            
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_HI.sub(' ', x))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_CTRL.sub(' ', x))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_URL.sub(' ', x))
        #self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_DIGIT.sub(' ', x))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_WS.sub(' ', x))        
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_NWC.sub(' ', x))        
        
    def cleanTextListItems(self, prefix=None):
        '''
        Cleans text data as list by:
        1.  force lowercase
        2.  remove non-ascii chars
        3.  standardize whitespace
        4.  remove digits
        5.  remove control characters
        6.  remove URL patterns
        '''
        self.data[prefix]   =   self.data[prefix].map(lambda x: '\n'.join(x).lower().decode('unicode_escape'))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_HI.sub(' ', x))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_CTRL.sub(' ', x))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_URL.sub(' ', x))
        #self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_DIGIT.sub(' ', x))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_WS.sub(' ', x))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_NWC.sub(' ', x))        
        
    def cleanTextDictItems(self, prefix=None):
        '''
        Cleans text data as list by:
        1.  unroll the values from a nesterd structure of lists
        2.  extract dict values and stack them as strings
        1.  force lowercase
        2.  remove non-ascii chars
        3.  standardize whitespace
        4.  remove digits
        5.  remove control characters
        6.  remove URL patterns
        '''
        self.data[prefix]   =   self.unrollDictValues(prefix)
        self.data[prefix]   =   self.data[prefix].map(lambda x: x.values() if isinstance(x, dict) else '')
        #self.data[prefix]   =   self.data[prefix].map(lambda x: '\n'.join(x).lower().decode('unicode_escape'))
        self.data[prefix]   =   self.data[prefix].map(lambda x: '\n'.join(x).lower().decode('utf-8'))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_HI.sub(' ', x))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_CTRL.sub(' ', x))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_URL.sub(' ', x))
        ##self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_DIGIT.sub(' ', x))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_WS.sub(' ', x))
        self.data[prefix]   =   self.data[prefix].map(lambda x: self.re_NWC.sub(' ', x))        

    def set_ngram_column_labels(self, prefix):
        self.name           = prefix.split('_')[0]
        self.name_tokens    = self.name+'_tokens'
        self.name_lemmatize = self.name+'_lemmatize'
        self.name_stemmed   = self.name+'_stemmed'
        self.name_bigrams   = self.name+'_bigrams'
        self.name_trigrams  = self.name+'_trigrams'
        self.name_fourgrams = self.name+'_fourgrams'
        
    def set_tokenized_column_labels(self, prefix):
        self.name           = prefix.split('_')[0]
        self.name_sentences = self.name+'_sentences'
        self.name_words     = self.name+'_words'
        self.name_stemmed   = self.name+'_stemmed'
        self.name_normalized= self.name+'_normalized'

    def blob(self, prefix, **kwargs):
        tokenizer   = kwargs['tokenizer']
        np_extract  = kwargs['np_extract']
        pos_tagger  = kwargs['pos_tagger']
        analyzer    = kwargs['analyzer']
        classifier  = kwargs['classifier']

        # generate Textblobs for line in pandas series
        blob = self.data[prefix].map(lambda l: TextBlob(l, tokenizer=tokenizer, np_extractor=np_extract, pos_tagger=pos_tagger, analyzer=analyzer, classifier=classifier))

        return blob
    
    def tokenize_sentences(self, prefix, **kwargs):
        tokenizer = kwargs['tokenizer']
        normalizer = kwargs['token_format']
        
        # tokenize the document into sentences from blob object
        sentences = self.data[prefix].map(lambda s: s.sentences)

        return sentences
    
    def tokenize_words(self, prefix, normalize = 'stem', **kwargs):
        tokenizer = kwargs['tokenizer']
        normalizer = kwargs['token_format']
        
        # tokenize each sentence into words
        # trim token whitespaces
        # eliminate tokens of character length 1
        words = self.data[prefix].map(lambda l: map(lambda w: w.strip().tokens, l))

        for wl in words:
            for w in wl:
                for t in w:
                    if not len(t)>1:
                        w.remove(t)

        return words    
        
    def normalize(self, prefix, **kwargs):
        tokenizer = kwargs['tokenizer']
        normalizer = kwargs['token_format']
        spelling = kwargs['spell_correct']
        
        # singularize tokens
        data = self.data[prefix].map(lambda l: map(lambda w: w.singularize(), l))
            
        # Spell correct flag
        # REALLY SHOULD NEVER BE USED
        if spelling:
            print("Spell Correction Invoked.....")
            data = data.map(lambda l: map(lambda wl: map(lambda w: w.correct(), wl), l))
            print(data.map(lambda l: map(lambda w: type(w), l)))

        # filter out 'bad' words, normalize good ones
        # w if w not in self.stopWords else wl.remove(w)
        data = data.map(lambda l: map(lambda wl: map(lambda w: wl.remove(w) if w in self.stopWords else w, wl), l))
        data = data.map(lambda l: map(lambda wl: map(lambda w: wl.remove(w) if w == '\'s' else w, wl), l))
        data = data.map(lambda l: map(lambda wl: map(lambda w: wl.remove(w) if w == '\'d' else w, wl), l))

        # remove tokens with length 1
        ree = re.compile(r'(\'\w)')
        rlen = len(data)
        tmp = data.copy()
        for index in range(0,rlen):
            wl_coll = list()
            for lst in tmp[index]:
                wl = list()
                for word in lst:
                    if not isinstance(word, types.NoneType):
                        if re.match(ree, word):
                            ree.sub('', word)
                        if len(word.strip().strip('.').strip(',')) > 1:
                            wl.append((word))
                wl_coll.append(WordList(wl))
            data[index] = wl_coll
        del tmp

        # remove via regexp c'c pattern

        # Stemming or lemmatization of tokens    
        if normalizer == 'stem':
            data = data.map(lambda l: map(lambda wl: map(lambda w: self.stemmer.stem(w) if w in wl and not isinstance(w, types.NoneType) else wl.remove(w), wl), l))
        elif normalizer == 'lemma':
            data = data.map(lambda l: map(lambda wl: map(lambda w: w.lemmatize(), wl), l))
        elif normalizer == 'None':
            pass

        data = data.map(lambda l: map(lambda wl: map(Word, wl), l))
        data = data.map(lambda l: map(WordList, l))

        return data
    
    def sentence_sentiment(self):
        if metric == 'polarity':
            return self.sentiment.polarity
        elif metric == 'subjectivity':
            return self.sentiment.subjectivity

    def ngrams(self, prefix, n=2):
        ngrams = self.data[prefix].map(lambda l: map(lambda wl: Sentence(' '.join(wl)).ngrams(n), l))

        if n>0:
            ngrams = ngrams.map(lambda l: map(lambda wl: map(lambda w: map(lambda i: i if not i.isnumeric() else w.remove(i), w), wl), l))
            ngrams = ngrams.map(lambda l: map(lambda wl: map(lambda w: filter(None, w), wl), l))

        ngrams = ngrams.map(lambda l: map(lambda wl: map(lambda n: ' '.join(n), wl), l))
        ngrams = ngrams.map(lambda l: list(itertools.chain(*l)))
        
        return ngrams

    def generate_ngrams(self, dataset, prefix):
        self.set_ngram_column_labels(prefix)
        
        dataset[self.name_tokens] = resumes.ngrams(prefix, n=1)
        print("Unigrams computed...")
        dataset[self.name_bigrams] = resumes.ngrams(prefix, n=2)
        print("Bigrams computed...")
        dataset[self.name_trigrams] = resumes.ngrams(prefix, n=3)
        print("Trigrams computed...")
        dataset[self.name_fourgrams] = resumes.ngrams(prefix, n=4)
        print("Fourgrams computed...")
        
        return None

    def tag_sentences(self, tokenized_sents):
        return None
    
    def chunk_tagged_sentences(self, tagged_sents):
        return None
    
    def get_chunks(self, chunk_type='NP'):
        return None


#   Initialize PandasNgram object to grab data from a Redis store and put it into a pandas data frame
sw = ['a', 'at', 'to', 'in', 'if', 'of', 'and', 'is', 'the', 'they', 'their', 'the', 'or', 'it', 'you', 'an', 'with', 'from', 'for', 'as', 'such']
resumes     = PandasNgram(hostname='plytos.com', DB=1, stop_words=None)

#   Grab the appropriate data
resumes.data['name_src']        =   resumes.getRedisData('name')
resumes.data['headline_src']    =   resumes.getRedisData('headline')
resumes.data['summary_src']     =   resumes.getRedisData('summary')
resumes.data['education_src']   =   resumes.getRedisData('education')
resumes.data['skills_src']      =   resumes.getRedisData('skills')
resumes.data['experience_src']  =   resumes.getRedisData('experience')

##   Clean and preprocess data
resumes.cleanText('name_src')
resumes.cleanText('headline_src')
resumes.cleanText('summary_src')
resumes.cleanTextDictItems('education_src')
resumes.cleanTextListItems('skills_src')
resumes.cleanTextDictItems('experience_src')

#tokenizers.WordTokenizer()
#nt.PunktSentenceTokenizer(),
tokenizer_prefs = {
    'tokenizer' : tokenizers.SentenceTokenizer(),
    'token_format' : 'stem',
    'spell_correct' : False,
    'np_extract': None,
    'pos_tagger': None,
    'analyzer': None,
    'classifier': None    
}

# Generate ngrams
# Drop digit unigrams after ngram creation
resumes.data
resumes.data['experience_blob'] = resumes.blob('experience_src', **tokenizer_prefs)
print("Blob Computed....")
resumes.data['experience_blob'] = resumes.tokenize_sentences('experience_blob', **tokenizer_prefs)
print("Sentences Tokenized....")
resumes.data['experience_words'] = resumes.tokenize_words('experience_blob', **tokenizer_prefs)
print("Words Tokenized....")
resumes.data['experience_norm'] = resumes.normalize('experience_words', **tokenizer_prefs)
print("Words normalized....")
resumes.generate_ngrams(dataset=resumes.ngram_collection, prefix='experience_norm')

print(resumes.ngram_collection.columns)

#   Get unique n-grams for lengths 1 through 4 inclusive
#   Combine separate n-gram lists into one large list with lengths increasing right to left in the list
#   Use this list of unique ngrams to generate n-gram feature columns

unique_unigrams     = set(itertools.chain(*resumes.ngram_collection['experience_tokens']))
unique_bigrams      = set(itertools.chain(*resumes.ngram_collection['experience_bigrams']))
unique_trigrams     = set(itertools.chain(*resumes.ngram_collection['experience_trigrams']))
unique_fourgrams    = set(itertools.chain(*resumes.ngram_collection['experience_fourgrams']))

unique_ngrams       = list(set(list(unique_unigrams) + list(unique_bigrams) + list(unique_trigrams) + list(unique_fourgrams)))

del unique_unigrams
del unique_bigrams
del unique_trigrams
del unique_fourgrams

#   Generate a zero-valued dataframe container with ngrams as features/columns

ngram_df    = pd.DataFrame(index=resumes.ngram_collection.index, columns=unique_ngrams)
ngram_df.fillna(value=0, inplace=True)
resumes.ngram_collection     = resumes.ngram_collection.join(ngram_df, how="outer")

del unique_ngrams
del ngram_df

#   Count up occurrences of each ngram in a job description by column label
columnList = resumes.ngram_collection.columns.tolist()

counter = 0
print("Computing Term Frequency Matrix...")
for ngram in columnList[5:]:
    index = columnList.index(ngram)
    nsize = len(resumes.ngram_collection.columns[index].split())
    counter+=1
    
    if nsize == 1:
        resumes.ngram_collection[ngram] = resumes.ngram_collection.experience_tokens.map(lambda word: word.count(ngram))
    elif nsize == 2:
        resumes.ngram_collection[ngram] = resumes.ngram_collection.experience_bigrams.map(lambda word: word.count(ngram))
    elif nsize == 3:
        resumes.ngram_collection[ngram] = resumes.ngram_collection.experience_trigrams.map(lambda word: word.count(ngram))
    elif nsize == 4:
        resumes.ngram_collection[ngram] = resumes.ngram_collection.experience_fourgrams.map(lambda word: word.count(ngram))

    print(counter)

print('Term Frequency Matrix Computed.')
print('Writing Term Frequency Matrix to CSV file....')
resumes.ngram_collection.to_csv('resume_term_freq_matrix3.csv', encoding='utf-8')
#resumes.ngram_collection.iloc[6:10].to_csv("/Volumes/Badr/Plytos/analytics/m2/resume_term_freq_matrix3.csv")
print('Completed CSV Export of Matrix')

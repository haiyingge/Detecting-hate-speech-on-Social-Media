# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer

# Input data files are available in the "../datahanlder/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#===========data preprocessing===============================================

#importing different libraries for analysis, processing and classification
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn import re #regular expression for text processing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer #word stemmer class
lemma = WordNetLemmatizer()

from nltk import FreqDist 
# vectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression #classification model
from sklearn.metrics import confusion_matrix, classification_report, f1_score # performance evaluation criteria

# Remove @user #hashtag stopword
def normalizer(tweet):
    tweets = " ".join(filter(lambda x: x[0]!= '@' , tweet.split()))
    tweets = re.sub('[^a-zA-Z]', ' ', tweets)
    tweets = tweets.lower()
    tweets = tweets.split()
    tweets = [word for word in tweets if not word in set(stopwords.words('english'))]
    tweets = [lemma.lemmatize(word) for word in tweets]
    tweets = " ".join(filter(lambda x: x[0]!= '#' , tweet.split()))
    return tweets

#===========================================================================
def extract_hashtag(tweet):
    tweets = " ".join(filter(lambda x: x[0]== '#', tweet.split()))
    tweets = tweets.lower()
    tweets = tweets.split()
    tweets = [lemma.lemmatize(word) for word in tweets]
    tweets = " ".join(tweets)
    return tweets.split()

#===========================================================================
"""
For each input tweet(id, context, label), 
    we will remove the @user, #hashtag and stopwords on context and generate clean data - called normalized_text
    we will extract hashtags - call hashtag(list of string)
"""
def preprocessdata(input_file, output_file):
    df1 = pd.read_csv(input_file, sep='\t')

    tweets = df1.tweet
    df1['normalized_text'] = tweets.apply(normalizer)
    df1['hashtag'] = tweets.apply(extract_hashtag)
    print(df1.head())
    df1.to_csv(output_file, sep='\t')

# ==========================================================================
def cal_word_frequency(input_file):
    def sortedDictValues(d): 
        items=d.items() 
        backitems=[[v[1],v[0]] for v in items] 
        backitems.sort()
        backitems = backitems[::-1]
        return [ (backitems[i][1],backitems[i][0]) for i in range(0,len(backitems))] 

    def word_counter(texts):
        cv = CountVectorizer()   
        cv_fit=cv.fit_transform(texts)    
        word_list = cv.get_feature_names();    
        count_list = cv_fit.toarray().sum(axis=0) 
        return sortedDictValues((dict(zip(word_list,count_list))))

    # importing training data
    df1 = pd.read_csv(input_file, sep='\t')
    # get data from all tweets
    t1 = df1['hashtag']
    # get Hatred data from Hatred tweets
    t2 = df1[df1['label']==1]['hashtag']
    
    # all tweets 
    all_words = " ".join(t1).split()
    # print(word_counter(all_words))

    all_words_fre_dict = word_counter(all_words)
    print(all_words_fre_dict[:10])

    #Hatred tweets
    hatred_words = " ".join(t2).split()
    hatred_words_fre_dict = word_counter(hatred_words)
    print(str(hatred_words_fre_dict[:15]).encode('utf-8'))
    # print(str(word_counter(hatred_words)).encode('utf-8'))
    # pass


# preprocessdata('../datahanlder/train.csv', './train.csv')
# preprocessdata('../datahanlder/test.csv', './test.csv')
cal_word_frequency('./train.csv')


#===========================================================================














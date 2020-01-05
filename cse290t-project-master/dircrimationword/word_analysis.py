# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import libsvm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import os
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
import pickle
from sklearn.externals import joblib

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

discrimation_words = [
    ('white', 92),
    ('racist', 79),
    ('black', 65),
    ('trump', 52),
    ('against', 49),
    ('libtard', 48),
    ('racism', 47),
    ('stomping', 32),
    ('hate', 31),
    ('woman', 29),
    ('obama', 29),
    ('girl', 25),
    ('race', 24),
    ('stop', 24),
    ('video', 24),
    ('america', 24),
    ('men', 24),
    ('sex', 23)]

result = []

train_data = pd.read_csv('../datapreprocessing/train.csv', sep='\t') 
test_data = pd.read_csv('../datapreprocessing/test.csv', sep='\t') 

filename = '../models/MultinomialNB.model'
def get_result(filename):
    result = []
    names = []
    values = []
    model = pickle.load(open(filename, 'rb'))
    prediction = model.predict(test_data.normalized_text.values.astype('U'))

    def substitute(y, word):
        return re.sub(r'(?:^|\W)(%s)(?:$|\W)' % word, '', str(y))

    for x in range(len(discrimation_words)):
        corrected_data = test_data.copy()
        # word_fre = corrected_data['normalized_text'].str.count(discrimation_words[x][0]).sum()
        corrected_data['normalized_text'] = corrected_data['normalized_text'].apply(lambda y: substitute(y, discrimation_words_test[x][0]))
        corrected_prediction = model.predict(corrected_data.normalized_text.values.astype('U'))
        dis_to_normal = 0
        normal_to_dis = 0
        for i in range(len(prediction)):
            if prediction[i] != corrected_prediction[i]:
                if corrected_prediction[i] == 1:
                    normal_to_dis+=1
                else:
                    dis_to_normal+=1

        result.append((discrimation_words[x][0],discrimation_words_test[x][1],dis_to_normal, dis_to_normal/discrimation_words_test[x][1]))
        if dis_to_normal/discrimation_words_test[x][1] > 0:
            names.append(discrimation_words[x][0])
            values.append(dis_to_normal/discrimation_words_test[x][1])
    return result, names, values
        
result, names, values = get_result(filename)
print(values)

import os

for dirpath, dirnames, filenames in os.walk('models'):
    for filename in filenames:
        if '.model' in filename:
            filename = 'models/' + filename
            result = get_result(filename)
            print(filename, result)

import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.figure(figsize=(20, 10))
plt.bar(names, values, color=['tomato', 'coral', 'orangered', 'darkorange', 'gold', 'yellowgreen', 'lightgreen', 'limegreen', 'mediumseagreen', 'darkturquoise', 'skyblue', 'cornflowerblue', 'slateblue', 'mediumpurple', 'violet', 'orchid'])
plt.ylabel("Hate Rate", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=30)
plt.show()







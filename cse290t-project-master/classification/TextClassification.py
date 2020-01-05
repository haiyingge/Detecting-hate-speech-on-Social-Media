import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

"""
Process:
1. load data
"""
class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


# load data
train_data = pd.read_csv('../datapreprocessing/train.csv', sep='\t') 
test_data = pd.read_csv('../datapreprocessing/test.csv', sep='\t') 

classifiers = [
    GaussianNB(),
    MultinomialNB(),
    LogisticRegression(),
    SGDClassifier(),
    DecisionTreeClassifier(),
    MLPClassifier(),
    # BernoulliRBM(),
    # KNeighborsClassifier(),
    # RadiusNeighborsClassifier(),
    SVC(),
    LinearSVC(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreeClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    # VotingClassifier()
]

results = {}

for model in classifiers:
    print(model)
    model_name = str(model).split('(')[0]
    clf = Pipeline([('tfidf', TfidfVectorizer(decode_error='replace', encoding='utf-8')),
                    ('to_dense', DenseTransformer()),
                    (model_name, model),])


    clf.fit(train_data.normalized_text.values.astype('U'), train_data.label.values)
    prediction = clf.predict(test_data.normalized_text.values.astype('U'))
    
    results[model_name] = classification_report(test_data.label, prediction, output_dict=True)
    print(results[model_name])

    f = open(f'../models/{model_name}.model', 'wb')
    pickle.dump(clf, f)

f = open('result.csv', 'w')
f.write('model\tprecision\trecall\tf1-score\taccuracy\n')
for m, r in results.items():
    f.write(f"{m}\t{r['macro avg']['precision']}\t{r['macro avg']['recall']}\t{r['macro avg']['f1-score']}\t{r['accuracy']}\n")
f.close()
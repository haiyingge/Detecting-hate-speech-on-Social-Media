import numpy as np
import nltk
import string
from nltk.corpus import stopwords
import gensim
import os
from sklearn.metrics.pairwise import cosine_similarity
import operator

stops = set(stopwords.words("english"))
punct = set(string.punctuation)


class Word2vecExtractor:

    def __init__(self, w2vecmodel):
        # w2v = gensim.models.Word2Vec.load_word2vec_format("nlp/tools/word2vec/trunk/GoogleNews-vectors-negative300.bin", binary=True)
        self.w2vecmodel = gensim.models.KeyedVectors.load_word2vec_format(w2vecmodel, binary=True)

    def sent2vec(self, sentence):
        #words = [word for word in nltk.word_tokenize(sentence) if word not in stops and word not in punct]
        words = str(sentence).split()
        res = np.zeros(self.w2vecmodel.vector_size)
        count = 0
        for word in words:
            if word in self.w2vecmodel:
                count += 1
                res += self.w2vecmodel[word]

        if count != 0:
            res /= count

        return res

    def word2v(self, word):
        res = np.zeros(self.w2vecmodel.vector_size)
        if word in self.w2vecmodel:
            res += self.w2vecmodel[word]
        return res


if __name__ == '__main__':
    W2vecextractor = Word2vecExtractor("/Users/geet/PycharmProjects/GoogleNews-vectors-negative300.bin")

    sentence = "Will the lack of females choosing STEM related careers create a new gender disparity for Gen-Z or is this set to change?"
    vec_rep = W2vecextractor.sent2vec(sentence)

    print(vec_rep)


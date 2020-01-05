
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import sys
import numpy as np
import pandas
import preprocess
import gensim
import codecs

def stemAndTokenize(text):
	'''
	:param text: the tweet text to be tokenized and stemmed
	:return: returns tokenized and stemmed text
	'''

	text = str(text)
	tokenizer = RegexpTokenizer(r'\w+')
	p_stemmer = PorterStemmer()
	tokens = tokenizer.tokenize(text)
	stemmed_tokens = [p_stemmer.stem(i) for i in tokens]
	return stemmed_tokens


def runLDA(df, numTopics, numWords, filename):
	'''
	:param df: the data, 
	:param numTopics: number of topics, 
	:param numWords: number of words to be returned for each topic
	:return: returns numWords number of words for each topic
	'''

	df['tweet'] = df['tweet'].apply(stemAndTokenize)
	texts = df.tweet.tolist()

	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=numTopics, id2word = dictionary, passes=20)

	results = ldamodel.print_topics(num_topics=numTopics, num_words=numWords)
	results_file = filename+"-LDA-k"+str(numTopics)+"-n"+str(numWords)+".txt"
	writeLDAResultstoFile(results, results_file)
	print("Results stored in file: "+results_file)
	
	return


def writeLDAResultstoFile(results, results_file):
	'''
	:param results: results returned from LDA, 
	:results_file: filename to which results are written
	'''
	fout = codecs.open(results_file, "w", encoding='utf-8')

	for i in range(0, len(results)):
		fout.write("Topic: "+str(i+1)+":\n")
		words = results[i][1].split("\" + ")
		for word in words:
			cleaned_word = word.split("*\"")[1]
			fout.write(cleaned_word+"\t")
		fout.write("\n\n")
	fout.close()	

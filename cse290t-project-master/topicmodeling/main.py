import pandas
import preprocess
import topicmodel
import sys


def getTopics(data, modelName, numTopics, numWords, filename):
    '''
    :param data: the data,
    :modelName: the topic modeling algorithm LDA
    :param numTopics: number of topics,
    :param numWords: number of words to be returned for each topic
    :return: returns numWords number of words for each topic
    '''
    return topicmodel.runLDA(data, numTopics, numWords, filename)

if __name__ == "__main__":

    PREPROCESSING = False
    #filename = "data/kaggle.csv"
    #filename = "data/davidson.csv"
    filename = "data/kaggle_davidson.csv"

    if(PREPROCESSING):
        df = preprocess.create_preprocessed_file(filename)
    else:
        with open(filename.split(".")[0] + "-preprocessed.csv", 'r') as nfh:
            df = pandas.read_csv(nfh)

    modelName = "LDA"
    #numTopics = 3
    #numWords = 10

    numTopics = input("Choose number of topics K:")
    numWords = input("Choose number of top words to display for each topic:")

    getTopics(df, modelName, int(numTopics), int(numWords), filename)

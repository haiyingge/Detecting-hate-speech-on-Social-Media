from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import sys
import pandas
import numpy
from preprocess import preprocess
from time import time
from word2vec import Word2vecExtractor
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import StandardScaler


def create_combined():
    '''
    :param filename: the input file containing tweets in a csv file with column named "Tweet_text" containing tweets. A file containing the preprocessed tweets is created, with same filename as input filename, but appended with -preprocessed
    '''

    df2 = pandas.DataFrame()
    filename = "data/davidson.csv"
    with open(filename, "r") as fh:
        df = pandas.read_csv(fh)

    for index, row in df.iterrows():
        if(row['class'] == 2):
            tweet_row = pandas.Series([row['tweet']])
            df2 = df2.append(tweet_row, ignore_index=True)

    filename = "data/kaggle.csv"
    with open(filename, "r") as fh:
        df = pandas.read_csv(fh)
    for index, row in df.iterrows():
         if (row['label'] == 1):
            tweet_row = pandas.Series([row['tweet']])
            df2 = df2.append(tweet_row, ignore_index=True)

    with open("kaggle_davidson-combined.csv", 'w') as nfh:
        df2.to_csv(nfh)


    print("Complete file created in: "+filename.split(".")[0]+"-combined.csv")
    return df

def cluster_data(features, k, X=None):
    '''
    :param features: features defined on the data to be clustered.
    :param k: Number of clusters desired
    '''

    #vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
    #X = vectorizer.fit_transform(features.values.astype('U'))

    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, verbose=True)
    print("Clustering sparse data with %s" % km)

    km.fit(features)

    d = {'tweet' : features, 'clusterID' : km.labels_}
    df = pandas.DataFrame(d)
    return df
	#return km.labels_

def create_cluster(sparse_data, nclust):
    # Manually override euclidean
    def euc_dist(X, Y=None, Y_norm_squared=None, squared=False):
        # return pairwise_distances(X, Y, metric = 'cosine', n_jobs = 10)
        return cosine_similarity(X, Y)

    k_means_.euclidean_distances = euc_dist

    print(sparse_data.shape)
    scaler = StandardScaler(with_mean=False)
    sparse_data = scaler.fit_transform(sparse_data)
    kmeans = k_means_.KMeans(n_clusters=nclust, n_jobs=20, random_state=3425)
    _ = kmeans.fit(sparse_data)
    return kmeans.labels_

def store_cluster_info(filename, df, k):
    '''
    :param filename: the input file containing preprocessed tweets. A file containing the cluster information is created, with same filename as input filename, with -preprocessed replaced by -clustered
    :param df: Dataframe containing the tweets and their cluster information
	'''
    with open("".join(filename.split("-")[:2])+"-clustered-"+str(k)+".csv", 'w') as nfh:
        for item in df:
            nfh.write(str(item)+"\n")
    print("File with clustering results created in: "+"".join(filename.split("-")[:2])+"-clustered.csv")
    return

if __name__ == "__main__":

    W2vecextractor = Word2vecExtractor("/Users/geet/PycharmProjects/GoogleNews-vectors-negative300.bin")

    #df = create_combined()

    with open("kaggle_davidson-combined.csv", "r") as fh:
        df = pandas.read_csv(fh)

    vec_rep = []
    for index, row in df.iterrows():
        vector = W2vecextractor.sent2vec(row['tweet'])
        vec_rep.append(vector.transpose())
    rep = numpy.array(vec_rep)
    print(rep.shape)


    k = input("Choose number of topics K:")
    #df = cluster_data(vec_rep, k)

    df = create_cluster(rep, int(k))
    filename = "data/kaggle_davidson_combined"
    store_cluster_info(filename, df, int(k))


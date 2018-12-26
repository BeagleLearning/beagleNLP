import math
import numpy as np
from sklearn.cluster import DBSCAN
from spacy.tokens import Doc
import document_distance as dd
Doc.set_extension("clusterLabel", default=-1)
EPSILON = 0.6
FRACTION_IN_CLUSTER = 0.1
STD_FRACTION = 1


def makeRepresentationMatrix(corpus):
    return np.array([doc._.vector for doc in corpus.documents])


def getEpsilonFromDistanceMatrix(corpus):
    return np.std(np.array(corpus.distanceMatrix)) * STD_FRACTION


def getMinSamples(corpus):
    nDocs = len(corpus.documents)
    return max(math.ceil(nDocs * FRACTION_IN_CLUSTER), 2)


def createClusterMap(corpus, clusteringAnalysis):
    clusters = {}
    for i in range(len(corpus.documents)):
        label = str(clusteringAnalysis.labels_[i])
        doc = corpus.documents[i]
        doc._.clusterLabel = label
        clusters[label] = clusters.get(label, []).append(doc._.tag or doc)
    corpus.clusters = clusters
    return corpus


def dbscan(corpus, eps=EPSILON, min_samples=None, metric="cosine"):
    if not min_samples:
        min_samples = getMinSamples(corpus)

    repMatrix = makeRepresentationMatrix(corpus)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    clustering = dbscan.fit(repMatrix)
    corpus = createClusterMap(corpus, clustering)
    printClusters(corpus)
    return corpus


def dbscanWithWordVectorDistances(corpus, eps=None, min_samples=None):
    corpus = dd.wordPairsDistanceMatrix(corpus)
    if not eps:
        eps = getEpsilonFromDistanceMatrix(corpus)
    if not min_samples:
        min_samples = getMinSamples(corpus)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    clustering = dbscan.fit(corpus.distanceMatrix)
    corpus = createClusterMap(corpus, clustering)
    printClusters(corpus)
    return corpus


def printClusters(corpus):
    clusters = {}
    for doc in corpus.documents:
        clusterList = clusters.get(str(doc._.clusterLabel), [])
        clusterList.append(doc.text)
        clusters[str(doc._.clusterLabel)] = clusterList

    print(clusters)

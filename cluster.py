import math
import numpy as np
from sklearn.cluster import DBSCAN
from spacy.tokens import Doc
import document_distance as dd
Doc.set_extension("clusterLabel", default=-1)
EPSILON = 0.6
FRACTION_IN_CLUSTER = 0.1
STD_FRACTION = 1.5


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
        labelGroup = clusters.get(label, [])
        labelGroup.append(doc)
        clusters[label] = labelGroup
    if "-1" in clusters:
        clusters["uncategorized"] = clusters["-1"]
        del clusters["-1"]
    corpus.clusters = clusters
    return corpus


def dbscan(corpus, eps=EPSILON, min_samples=None, metric="cosine"):
    if not min_samples:
        min_samples = getMinSamples(corpus)

    repMatrix = makeRepresentationMatrix(corpus)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    clustering = dbscan.fit(repMatrix)
    corpus = nameClusters(createClusterMap(corpus, clustering))
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
    corpus = nameClusters(createClusterMap(corpus, clustering))
    printClusters(corpus)
    return corpus


def printClusters(corpus):
    clusters = {}
    for doc in corpus.documents:
        clusterList = clusters.get(str(doc._.clusterLabel), [])
        clusterList.append(doc.text)
        clusters[str(doc._.clusterLabel)] = clusterList

    print(clusters)


def findClusterCentroid(cluster):
    numDocs = len(cluster)
    if numDocs == 0:
        return None
    else:
        return np.sum([doc._.vector for doc in cluster], axis=0) / numDocs


def nameClusters(corpus):
    # each cluster needs to get some sort of title.
    # so I think we should extract all keywords
    # then for each keyword, we look at which single one is closest to the
    # center of the cluster.
    # smallest average distance
    taggedClusters = {}
    for (tag, cluster) in corpus.clusters.items():
        if tag == "uncategorized":
            taggedClusters["uncategorized"] = cluster
            continue
        centroid = findClusterCentroid(cluster)
        keywords = [t for doc in cluster for t in doc]
        keywordVecs = np.array([k.vector for k in keywords])
        dists = np.linalg.norm(keywordVecs - centroid)
        minIndex = np.argmin(dists)
        print(f"New keyword is {keywords[minIndex].text}")
        taggedClusters[keywords[minIndex].text] = cluster
    corpus.clusters = taggedClusters
    return corpus

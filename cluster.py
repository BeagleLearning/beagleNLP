import math
from beagleError import BeagleError
import errors
import numpy as np
from sklearn.cluster import DBSCAN
from spacy.tokens import Doc
import document_distance as dd
import analysis
Doc.set_extension("clusterLabel", default=-1, force=True)
EPSILON = 0.25
FRACTION_IN_CLUSTER = 0.05
STD_FRACTION = 1.0

import analysis


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
    if math.isnan(eps) or eps <= 0:
        raise BeagleError(errors.NO_WORD_DATA)
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


def nameClusters(corpus, algorithm="textrank"):
    # each cluster needs to get some sort of title.
    # so I think we should extract all keywords
    # then for each keyword, we look at which single one is closest to the
    # center of the cluster.
    # smallest average distance
    taggedClusters = {}
    for (tag, cluster) in corpus.clusters.items():
        keyword = ""
        if tag == "uncategorized":
            taggedClusters["uncategorized"] = cluster
            continue
        if algorithm == "textrank":
            keyword = analysis.textrank_keywords([q.text for q in cluster], corpus)
            print(f"\n New keyword text {keyword} \n")
        else:
            centroid = findClusterCentroid(cluster)
            keywords = [t for doc in cluster for t in doc if not t.is_oov and not t.is_stop and not t.is_punct and len(t.text.strip()) > 1]
            keywordVecs = np.array([k.vector for k in keywords])
            vecToCentroid = keywordVecs - centroid
            dists = np.linalg.norm(vecToCentroid, axis=1)
            print(f"dists {[[dists[i], keywords[i].text] for i in range(len(dists))]}")
            minIndex = np.argmin(dists)
            print(f"New keyword text {keywords[minIndex].text}")
            keyword = keywords[minIndex].text
        taggedClusters[keyword] = cluster
    corpus.clusters = taggedClusters
    return corpus

import math
from beagleError import BeagleError
import errors
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor
from spacy.tokens import Doc
import document_distance as dd
import analysis
import gaussian_kmeans
import textrank
import time
Doc.set_extension("clusterLabel", default=-1, force=True)
EPSILON = 0.25
FRACTION_IN_CLUSTER = 0.05
STD_FRACTION = 1.0


def makeRepresentationMatrix(corpus):
    return np.array([doc._.vector for doc in corpus.documents])


def getEpsilonFromDistanceMatrix(corpus):
    return np.std(np.array(corpus.distanceMatrix)) * STD_FRACTION


def getMinSamples(corpus):
    nDocs = len(corpus.documents)
    return max(math.ceil(nDocs * FRACTION_IN_CLUSTER), 2)


def createClusterMap(corpus, clusteringAnalysis):
    clusters = {}
    # for i in range(len(corpus.documents)):
    for i in range(len(clusteringAnalysis.labels_)):
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
    print(f"Clusters: {clusters}")#JEFFLAG
    #Perhaps need to add a check: if we didn't create uncategorized, ensure it exists.
    return corpus


def getOutliers(noisy_data, n_neighbors=5, corpus=None):
    """
    Using the Local Outlier Factor method, identify the outliers and remove them from the provided data. This functions assumes at least 5% contamination and uses the minkowski metric with a p-value of 1.5 (between manhattan and eucledian).

    Parameters:
        noisy_data (nD array): Multidimensional array which is a matrix that represents the set of documents to be clustered.
        n_neighbors (int): number of neighbors to examine when determining its "outlier" factor
        corpus (corpus): Optional parameter for debugging purposes, used to log out the outliers, and can be used for tracking purposes in prod.
            (default is None)

    Returns:
        outlier_indexes (list): A list of indeces of the outlier quetions
        clean_data (list):
    """

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric="minkowski", p=1.5, contamination=0.05)
    pred = lof.fit_predict(noisy_data)
    outlier_indexes = np.where(pred == -1)[0]
    mask = np.ones(len(noisy_data), dtype=bool)
    mask[outlier_indexes] = False
    if corpus is not None:
        outliers = [corpus.documents[index] for index in outlier_indexes]
        # print(outliers)
    clean_data = noisy_data[mask]
    return outlier_indexes, clean_data


def findBestFitCluster(orphanCorpus, corpusCluster={}):
    """
    Given a set of questions without a cluster and a set of other clusters, find the best cluster to put the orphaned questions
    Parameters:
        orphanCorpus (tagged_question_corpus.TaggedQuestionCorpus): corpus of the questions without a cluster.
        corpusCluster ({tagged_question_corpus.TaggedQuestionCorpus}): Object containing different clusters and their corpuses

    Returns:
        xxx
    """

    # corpusCluster = {
    #     "questions": [ 'and the moon too guys', 'lets show some or a lot of love for the moon!!' ],
    #     "question_vectors": [[], []],
    #     "clusterIds": [ '4', '4' ]
    # }

    # orphanCorpus = [ {
    #         "id": 11, "question": 'Another one about the sun?', "question_vector": []
    #     },
    #     {
    #         "id": 33,
    #         "question": 'What is the distance from the sun though?', "question_vector": [] },
    #     {
    #         "id": 37,
    #         "question": 'what\'s the changing factors of the sun and moon together?', "question_vector": []
    # } ]


    # Fit the Naive bayes model on existing clusters
    clf = ComplementNB()
    clf.fit(corpusCluster["question_vectors"], corpusCluster["clusterIds"])

    predictions = clf.predict_proba([doc["question_vector"] for doc in orphanCorpus])
    # print(predictions)


def agglomerate(corpus, threshold=1.4, ignoreOutliers=True):
    """
    Cluster a set of questions using the hierarchical (bottom-up) agglomerative clustering method.

    Parameters:
        corpus (list): Tagged Questions Corpus collection of questions and their ids
        threshold (int): Interger value to determine the distance threshold for the cut-off point as we build the dendrogram
        removeOutliers (bool): A flag to determine whether to remove outliers or not
            (default is True)

    Returns:
        corpus: Corpus that has clusters list attached to it
    """

    repMatrix = makeRepresentationMatrix(corpus)

    outliers = []
    if ignoreOutliers:
        outliers, repMatrix = getOutliers(repMatrix, corpus=corpus)

    clustering = AgglomerativeClustering(
        linkage="ward",
        distance_threshold=threshold,
        n_clusters=None
    )

    clustering.fit(repMatrix)
    mapping = [i[1] - i[0] for i in enumerate(outliers)]
    clustering.labels_ = np.insert(clustering.labels_,  mapping, -1)
    clusterMap = createClusterMap(corpus, clustering)
    print(clusterMap)#JEFFLAG
    corpus = nameClusters(clusterMap)
    print("The corpus's clusters after naming")
    print(corpus.clusters)#JEFFLAG
    return corpus


def g_kmeans(corpus, strictness=4, ignoreOutliers=True):
    """
    Gaussian "K" K-means clustering. The number of clusters is determined by a top-down (divisive) agglomerative method.

    Parameters:
        corpus (list): Tagged Questions Corpus collection of questions and their ids
        strictness (int): >= 0 but < 5, this value determins the strictness of the anderson         statistical test
        removeOutliers (bool): A flag to determine whether to remove outliers or not
            (default is True)

    Returns:
        corpus: Corpus that has clusters list attached to it

    Reference:
        https://papers.nips.cc/paper/2526-learning-the-k-in-k-means.pdf
    """

    repMatrix = makeRepresentationMatrix(corpus)
    outliers = []
    if ignoreOutliers:
        outliers, repMatrix = getOutliers(repMatrix, corpus=corpus)
    clustering = gaussian_kmeans.get_clusters(repMatrix, strictness, outliers)
    cluserMap = createClusterMap(corpus, clustering)
    corpus = nameClusters(cluserMap)
    printClusters(corpus)
    return corpus


def kmeans(corpus, n_clusters=5, ignoreOutliers=True):
    """
    Clusters questions into the number of clusters required.

    Parameters:
        corpus (list): Tagged Questions Corpus collection of questions and their ids
        n_clusters (int): The number of clusters to return
        removeOutliers (bool): A flag to determine whether to remove outliers or not
            (default is True)

    Returns:
        corpus: Corpus that has clusters list attached to it
    """

    repMatrix = makeRepresentationMatrix(corpus)
    outliers = []
    if ignoreOutliers:
        outliers, repMatrix = getOutliers(repMatrix, corpus=corpus)
    kmeans = KMeans(random_state=0, n_clusters=n_clusters, n_init=10,
                    max_iter=300, tol=1e-4, precompute_distances="auto")
    clustering = kmeans.fit(repMatrix)
    mapping = [i[1] - i[0] for i in enumerate(outliers)]
    clustering.labels_ = np.insert(clustering.labels_,  mapping, -1)
    cluserMap = createClusterMap(corpus, clustering)
    corpus = nameClusters(cluserMap)
    # printClusters(corpus)
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
    print("\n we have {} clusters".format(len(corpus.clusters.items())))
    for (tag, cluster) in corpus.clusters.items():
        keyword = ""
        if tag == "uncategorized":
            taggedClusters["uncategorized"] = cluster
            continue
        if algorithm == "textrank":
            keyword = textrank.get_keywords(cluster, corpus)
            print(f"\n New keyword text {keyword} for cluster #{tag} \n")
        else:
            centroid = findClusterCentroid(cluster)
            keywords = [t for doc in cluster for t in doc if not t.is_oov and not t.is_stop and not t.is_punct and len(
                t.text.strip()) > 1]
            keywordVecs = np.array([k.vector for k in keywords])
            vecToCentroid = keywordVecs - centroid
            dists = np.linalg.norm(vecToCentroid, axis=1)
            print(
                f"dists {[[dists[i], keywords[i].text] for i in range(len(dists))]}")
            minIndex = np.argmin(dists)
            print(f"New keyword text {keywords[minIndex].text}")
            keyword = keywords[minIndex].text
        if keyword in taggedClusters:
            print(taggedClusters[keyword])
            # print(cluster)
            taggedClusters[keyword] = taggedClusters[keyword] + cluster
            # taggedClusters[keyword] = np.concatenate(
            # [taggedClusters[keyword], cluster])
        else:
            taggedClusters[keyword] = cluster
    corpus.clusters = taggedClusters
    corpus = groupSmallClusters(corpus, 1)
    return corpus


def groupSmallClusters(corpus, max_cluster_size=1):
    """
    Takes all small clusters, given a threshold, and merges them with uncategorized. This is essential for agglomerative clustering and gaussian kmeans where each individual question is considered its own group.

    Parameters:
        corpus (list): Tagged Questions Corpus collection of questions and their clusters
        max_cluster_size (int): The maximum number of questions in a cluster to determine whether its uncategorized or not.
            (default is 1)

    Returns:
        corpus: Corpus with small clusters merged into the uncategorized cluster
    """

    smallClusterNames = []
    corpus.clusters["uncategorized"] = corpus.clusters.get("uncategorized", [])
    for c in corpus.clusters:
        if len(corpus.clusters[c]) <= max_cluster_size:
            corpus.clusters["uncategorized"] = corpus.clusters["uncategorized"] + \
                corpus.clusters[c]
            smallClusterNames.append(c)

    for name in smallClusterNames:
        corpus.clusters.pop(name)

    return corpus

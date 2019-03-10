# pairs docOne words to closest docTwo word
import numpy as np
from functools import partial
import math

DISTANCE_METRICS = ["cosine", "euclidean", "spacy"]


def createDistanceMatrix(corpus, distFunction):
    nDocs = len(corpus.documents)
    corpus.distanceMatrix = np.ones((nDocs, nDocs))
    for i in range(nDocs):
        docOne = corpus.documents[i]
        for j in range(i, nDocs):
            docTwo = corpus.documents[j]
            dist = distFunction(docOne, docTwo)
            corpus.distanceMatrix[i][j] = dist
            corpus.distanceMatrix[j][i] = dist
    return corpus


def distToClosestTokenInDoc(token, comparisonDoc, metric):
    vec = token.vector
    comparisonVectors = np.array([t.vector for t in comparisonDoc])

    if metric == "cosine":
        norms = np.linalg.norm(comparisonVectors, axis=1) * np.linalg.norm(vec)
        norms[norms == 0.0] = 1.0
        dot = np.dot(comparisonVectors, vec)
        dotText = [[comparisonDoc[i].text, dot[i]] for i in range(len(comparisonDoc))]
        print(f"dot: {dotText}")
        dists = np.abs(dot / norms)
        minInd = np.argmax(dists)
        print(f"{token} matched with {comparisonDoc[minInd].text}")
        maxDist = np.max(dists)
        print(f"final dist {1 - maxDist}")
        return 1 - maxDist
    elif metric == "euclidean":
        dists = np.linalg.norm(comparisonVectors - vec, axis=1)
        # minInd = np.argmin(dists)
        minDist = np.min(dists)
        return minDist
    else:
        dists = [token.similarity(t) for t in comparisonDoc]
        # maxInd = np.argmax(dists)
        minDist = 1 - max(dists)
        return minDist


def directionalWordPairsDistance(doc, comparison, metric):
    # for each token (lemma?) in doc One
    # check distance to each token (lemma?) in doc two
    # choose closest other word
    # calculate euclidean distance between them
    # weight by tfidf of doc one term
    total = 0
    lemmasProcessed = []
    numWordsHandled = 1
    for token in doc:
        # if this lemma has not already been encountered, then handle it
        if (token.lemma_ not in lemmasProcessed
                and not token.is_stop
                and not token.is_oov):
            dist = distToClosestTokenInDoc(token, comparison, metric)
            # + 1 to avoid issues with tfidf being 0 (all docs have word)
            tfidf = 1 + doc._.globalTfidf[token.lemma_]
            update = dist * tfidf
            total = total + update
            # record how many words we have considered for scaling purposes
            numWordsHandled = (numWordsHandled
                               + doc._.textFrequency[token.lemma_])
            # skip other words with same lemma in future - already handled with
            # tfidf count
            lemmasProcessed.append(token.lemma_)

    return total / numWordsHandled


def wordPairsDistanceForDocs(docOne, docTwo, metric="cosine"):
    return (directionalWordPairsDistance(docOne, docTwo, metric)
            + directionalWordPairsDistance(docTwo, docOne, metric)) / 2.0


#
# DISTANCE MATRIX FUNCTIONS
#


def wordPairsDistanceMatrix(corpus, metric="cosine"):
    distFunction = partial(wordPairsDistanceForDocs, metric=metric)
    return createDistanceMatrix(corpus, distFunction)


def simpleDistanceMatrix(corpus):
    return createDistanceMatrix(corpus, lambda one, two: one.similarity(two))

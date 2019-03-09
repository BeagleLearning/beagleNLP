import errors
from beagleError import BeagleError
import numpy as np
from spacy.tokens import Doc
Doc.set_extension("vector", default={})
# create document vectors


def findVectorLength(corpus):
    for doc in corpus.documents:
        if len(doc) > 0:
            for token in doc:
                if len(token.vector) > 0:
                    return len(token.vector)
    return 0


def tfidfWeightedBagOfWords(corpus, withStops=False, useGlobalIDF=False):
    # NOTE: how could we use broader frequency information to our benefit?

    # figure out how long the vectors we are using are
    length = findVectorLength(corpus)
    if length == 0:
        print("No vectors found when calculating bag-of-words vectors")
        raise BeagleError(errors.NO_VECTORS_FOUND)

    for doc in corpus.documents:
        doc._.vector = np.zeros(length)
        tfidfDict = None
        if useGlobalIDF:
            tfidfDict = doc._.globalTfidf
        else:
            tfidfDict = doc._.tfidf
        # TODO: consider if there is a faster way of doing this if we vectorize
        for token in doc:
            if token.is_stop and not withStops:
                continue
            else:
                # adding 1 avoids issues when a word is in every doc,
                # and therefore tfidf = 0
                tfidf = 1 + tfidfDict[token.lemma_]
                doc._.vector = doc._.vector + (token.vector * tfidf * tfidf * tfidf)

        # normalize
        norm = np.linalg.norm(doc._.vector)
        if norm != 0:
            doc._.vector = doc._.vector / norm

    return corpus


def tfidfWeightedVerbAndNoun(corpus, useGlobalIDF=False):
    # figure out how long the vectors we are using are
    length = findVectorLength(corpus)
    if length == 0:
        raise BeagleError(errors.NO_VECTORS_FOUND)
    for doc in corpus.documents:
        tfidfDict = None
        if useGlobalIDF:
            tfidfDict = doc._.globalTfidf
        else:
            tfidfDict = doc._.tfidf
        doc._.vector = np.zeros(length)
        # TODO: consider if there is a faster way of doing this if we vectorize
        for token in doc:
            tfidf = 1 + tfidfDict[token.lemma_]
            tokVec = token.vector * tfidf
            if token.pos_ == "VERB" or token.pos_ == "NOUN" or token.pos_ == "PROPN":
                doc._.vector = doc._.vector + tokVec

    return corpus

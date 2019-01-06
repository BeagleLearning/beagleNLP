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


def tfidfWeightedBagOfWords(corpus, withStops=False):
    # NOTE: how could we use broader frequency information to our benefit?

    # figure out how long the vectors we are using are
    length = findVectorLength(corpus)
    if length == 0:
        raise BeagleError(errors.NO_VECTORS_FOUND)

    for doc in corpus.documents:
        doc._.vector = np.zeros(300)
        # TODO: consider if there is a faster way of doing this if we vectorize
        for token in doc:
            if token.is_stop and not withStops:
                continue
            else:
                # adding 1 avoids issues when a word is in every doc,
                # and therefore tfidf = 0
                tfidf = 1 + doc._.tfidf[token.lemma_]
                doc._.vector = doc._.vector + (token.vector * tfidf)

        # normalize
        norm = np.linalg.norm(doc._.vector)
        if norm != 0:
            doc._.vector = doc._.vector / norm

    return corpus


def tfidfWeightedVerbAndNoun(corpus):
    # figure out how long the vectors we are using are
    length = findVectorLength(corpus)
    if length == 0:
        raise BeagleError(errors.NO_VECTORS_FOUND)

    verb = np.zeros(length)
    noun = np.zeros(length)
    for doc in corpus.documents:
        doc._.vector = np.zeros(300)
        # TODO: consider if there is a faster way of doing this if we vectorize
        for token in doc:
            tfidf = 1 + doc._.tfidf[token.lemma_]
            tokVec = token.vector * tfidf
            if token.pos_ == "VERB":
                verb = verb + tokVec
            elif token.pos_ == "NOUN" or token.pos_ == "PROPN":
                noun = noun + tokVec

    # combine
    doc._.vector = np.concatenate((verb, noun))

    return corpus

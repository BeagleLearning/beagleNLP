#
# There are broadly two ways to cluster documents:
# 1. Calculate document distances, then run distance-based clustering
# 2. Create vector document representations, then run clustering

# We support the following ways of calculating document distances:
# 1. Spacy similarity measure (based on average of token vectors)
# 2. Word-pairs distance. Total dist is average dist of closest word pairs
#    Note that we support three distance metrics (cosine, euclidean, spacy)

# We support the following ways of generating document representations
# 1. tfidf weighed back of words
# 2. tfid weighted back-of-verbs plus back-of-nouns
#

import numpy as np
import spacy
from errors import UNKNOWN_KEYWORDS
from beagleError import BeagleError
from corpus import Corpus
from tagged_question_corpus import TaggedQuestionCorpus
import tfidf as tfidf
import document_vector as docVec
import cluster
from cluster_on_keywords import clusterOnKeywords


# nlp = spacy.load("./model/googlenews-spacy");
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe(tfidf.generateTextFrequency, name="text_frequency")


def buildCorpus(docs):
    corpus = Corpus(docs, nlp)
    corpus = tfidf.tfidf(corpus)
    return docVec.tfidfWeightedBagOfWords(corpus, useGlobalIDF=True)


def clusterDocuments(docs):
    corpus = buildCorpus(docs)
    return cluster.dbscanWithWordVectorDistances(corpus)


def buildTaggedCorpus(docs):
    corpus = TaggedQuestionCorpus(docs, nlp)
    corpus = tfidf.tfidf(corpus)
    return docVec.tfidfWeightedBagOfWords(corpus, useGlobalIDF=True)


def clusterQuestions(docs):
    corpus = buildTaggedCorpus(docs)
    return cluster.dbscanWithWordVectorDistances(corpus)


def clusterQuestionsOnKeywords(questions, keywords):
    questionsCorpus = buildTaggedCorpus(questions)
    keywordsCorpus = buildCorpus(keywords)
    #removed = keywordsCorpus.removeUnknownDocs()
    #if removed:
    #   raise BeagleError(UNKNOWN_KEYWORDS)
    #else:
    return clusterOnKeywords(questionsCorpus, keywordsCorpus)


def dist(wordOne, wordTwo):
    one = nlp(wordOne)
    two = nlp(wordTwo)
    if (np.linalg.norm(one.vector) == 0
       or np.linalg.norm(two.vector) == 0):
        raise Exception("Don't have both words")
    return one.similarity(two)


def getVector(word):
    return nlp(word).vector

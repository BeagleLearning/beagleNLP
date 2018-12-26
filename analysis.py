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
    return docVec.tfidfWeightedBagOfWords(corpus)


def clusterDocuments(docs):
    corpus = buildCorpus(docs)
    return cluster.dbscanWithWordVectorDistances(corpus)


def buildTaggedCorpus(docs):
    corpus = TaggedQuestionCorpus(docs, nlp)
    corpus = tfidf.tfidf(corpus)
    return docVec.tfidfWeightedBagOfWords(corpus)


def clusterQuestions(docs):
    corpus = buildTaggedCorpus(docs)
    return cluster.dbscanWithWordVectorDistances(corpus)


def clusterQuestionsOnKeywords(questions, keywords):
    questionsCorpus = buildTaggedCorpus(questions)
    keywordsCorpus = buildCorpus(keywords)
    return clusterOnKeywords(questionsCorpus, keywordsCorpus)


def dist(wordOne, wordTwo):
    one = nlp(wordOne)
    two = nlp(wordTwo)
    if (np.linalg.norm(np.array(one.vector)) == 0
       or np.linalg.norm(np.array(two.vector)) == 0):
        raise Exception("Don't have both words")
    return one.similarity(two)


def getVector(word):
    return nlp(word).vector

# a few things we want to try:
# 1) given a set of keywords, cluster the questions around those keywords
# 2) given a set of questions, cluster based on all words
# 3) given a set of questions, cluster based on nouns
# 4) given a set of questions, cluster based on nounds and verbs, concatenated
# 5a) given a set of questions, select important keywords (using wordfreq?)
# 5b) cluster the keywords to form topics
# 5c) cluster questions based on where their keywords lie ("topics")
# 6) given a set of questions, cluster strictly based on comon keywords, selecting
#    a representative group of keywords somehow...

# API definition:
# keywords + questions => clustering around keywords
# questions => suggested clustering, with tags

# ideally these should both work by the same mechanism.
# in other words, they should come up with the same clusters.

# so how would we cluster around a keyword?
# we'd get that keyword's vector, and then
# check the distance from that keyword to the sum of the question words

# so then if we were going to do that backwords... we are going to start with the questions...
# For each keyword in all questions, cluster around them
# We want to find a set of keywords that are good representations of the questions

# so we are going to start with that. Let's build a clustering system
# around the concept of tfidf weighted vectors

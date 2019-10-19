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
from collections import OrderedDict
import regex as re
from errors import UNKNOWN_KEYWORDS
from beagleError import BeagleError
from corpus import Corpus
from tagged_question_corpus import TaggedQuestionCorpus
from build_tag_cluster import buildTagCluster
import tfidf as tfidf
import document_vector as docVec
import cluster
import textrank
import time
from spacy.tokens import Doc
from sklearn.feature_extraction.text import TfidfVectorizer

from cluster_on_keywords import clusterOnKeywords, findKeywordClusters
# Doc.set_extension("vector", default={})
# Doc.set_extension("vocabulary_", default={})
# Doc.set_extension("idf_", default={})
#
#
# # nlp = spacy.load("./model/googlenews-spacy");
# nlp = spacy.load("en_core_web_lg")
# nlp.add_pipe(tfidf.generateTextFrequency, name="text_frequency")
# # Possibly remove the leading statements that might help imply meaning of question
# # nlp.Defaults.stop_words -= {"what", "when", "who", "where", "why", "how"}
#
#
# def clean_text(text, lower=False):
#     """
#     Remove special characters
#
#     Parameters:
#         text (str): The string to be cleaned
#         lower (bool): A flag to determine whether to lower case the text or not
#             (default is False)
#
#     Returns:
#         string: a string that is free of special/punctuation characters
#     """
#     text = text.lower() if lower else text
#     text = text.replace("'", "")
#     text = re.sub("&lt;/?.*?&gt;", "&lt;&gt;", text)
#     return re.sub("[^\w ]", "", text)
#
#
# def buildCorpus(docs):
#     corpus = Corpus(docs, nlp)
#     corpus = tfidf.tfidf(corpus)
#     return docVec.tfidfWeightedBagOfWords(corpus, useGlobalIDF=True)
#
#
# def clusterDocuments(docs):
#     corpus = buildCorpus(docs)
#     return cluster.dbscanWithWordVectorDistances(corpus)
#
#
# def buildTaggedCorpus(docs):
#     corpus = TaggedQuestionCorpus(docs, nlp)
#     corpus = tfidf.tfidf(corpus)
#     return docVec.tfidfWeightedBagOfWords(corpus, useGlobalIDF=True)
#
#
# def tagAndVectorizeCorpus(docs):
#     """
#     Builds a tagged questions corpus and TFIDF vectorizes it.
#
#     Parameters:
#         docs (list): List of strings (questions)
#
#     Returs:
#         corpus (TaggedQuestionCorpus): returns a tagged corpus with with .idf_ and .vocabulary_         properties.
#
#     References:
#         TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#     """
#
#     corpus = TaggedQuestionCorpus(docs, nlp)
#     vectorizer = TfidfVectorizer(
#         analyzer='word',
#         ngram_range=(1, 3),
#         use_idf=True,
#         sublinear_tf=True,
#         smooth_idf=True,
#         stop_words='english'
#     )
#     vectors = vectorizer.fit_transform([doc._.lemmatized for doc in corpus.documents]).toarray()
#
#     corpus.vocabulary_ = vectorizer.vocabulary_
#     corpus.idf_ = vectorizer.idf_
#
#     for ix, doc in enumerate(corpus.documents):
#         doc._.vector = vectors[ix]
#
#     return corpus
#
#
# def customClusterQuestions(docs, algorithm, parameters, removeOutliers=True):
#     """
#     Customized question clustering methods. Implemented as a knock-off to prevent interupting with production builds
#
#     Parameters:
#         docs (list): List of strings (questions)
#         algorithm (str):  The clustering algorithm that will be used
#             (DBSCAN, K Means, Gaussian K Means, Agglomerative Clustering)
#         parameters (string): Parameters used to initiate the clustering algorithms
#             (epsilon, clusters, strictness, threshold)
#         removeOutliers (bool): A flag to determine whether to remove outliers or not
#             (default is True)
#
#     Returns:
#         corpus: Corpus that has clusters list attached to it
#     """
#
#     corpus = tagAndVectorizeCorpus(docs)
#     params = [{param["param"]: param["value"]} for param in parameters][0]
#     if algorithm == "DBSCAN":
#         return cluster.dbscan(corpus, float(params["epsilon"]))
#
#     if algorithm == "K Means":
#         return cluster.kmeans(corpus, int(params["clusters"]), removeOutliers)
#
#     if algorithm == "Gaussian K Means":
#         return cluster.g_kmeans(corpus, min(4, int(params["strictness"])), removeOutliers)
#
#     if algorithm == "Agglomerative Clustering":
#         return cluster.agglomerate(corpus, float(params["threshold"]), removeOutliers)
#
#
# def clusterQuestions(docs):
#     corpus = tagAndVectorizeCorpus(docs)
#     return cluster.agglomerate(corpus)
#
#
# def clusterQuestionsOnKeywords(questions, keywords, do_agglomeration):
#     """
#     A three step process to find the clusters in a set of questions when some keywords are provided. We are assuming that the list of keywords is always non-exhaustive and so we are expecting the keywords provided to not cover the entire corpus.
#
#     Step 1:
#         a. Use the complement Naive Bayes algorithm to determine which questions have the highest       probability of generating each keyword - the higher the probability, the more likely        the keyword and the question are in the same cluster.
#         b. If one question is likely to come from 2 separate clusters, pick the cluster with the        highest probabilty.
#         c. Return the clusters and the list of questions that do not belong to any of these             clusters.
#
#     Step 2:
#         a. Using the preffered clustering algorithm (agglomarative clustering), cluster the             remaining unclustered questions.
#         b. Name the clusters identified and return the question clusters
#
#     Step 3:
#         a. Find any 2 clusters that share the same keyword and merge those clusters into one.
#
#     Parameters:
#         questions (list[object]): A list of objects with the id (key) and the question (value)          string
#         keywords (list[string]): A list of keyword strings.
#             **Note: We are assuming the keywords provided are the ones generated or accepted by the instructor. keywords generated by the algorithm in the past are not expected**
#
#     Returns:
#         tagged_question_clusters (object{keyword: list[qnId]}): An object with the keyword (key)        and a list of question ids(value) that belong to that keyword.
#     """
#     keyword_clusters, uncategorized_questions = findKeywordClusters(questions, keywords)
#     print("\n\n Keyword Clusters", keyword_clusters)
#     tagged_question_clusters = {}
#     if len(uncategorized_questions) > 0:
#         if do_agglomeration:
#             question_clusters = clusterQuestions(uncategorized_questions)
#             tagged_question_clusters = buildTagCluster(question_clusters)
#         else:
#             tagged_question_clusters["uncategorized"] = [qn["id"] for qn in uncategorized_questions]
#
#     # Consolidate the two cluster objects, looking for clusters with the same keyword
#     for key in keyword_clusters:
#         if key in tagged_question_clusters.keys():
#             tagged_question_clusters[key] += keyword_clusters[key]
#         else:
#             tagged_question_clusters[key] = keyword_clusters[key]
#
#     return tagged_question_clusters
#
#     # questionsCorpus = buildTaggedCorpus(uncategorized_questions)
#     # keywordsCorpus = buildCorpus(keywords)
#     #removed = keywordsCorpus.removeUnknownDocs()
#     # if removed:
#     #   raise BeagleError(UNKNOWN_KEYWORDS)
#     # else:
#     # return clusterOnKeywords(questionsCorpus, keywordsCorpus)
#
#
# def matchQuestionsWithCategories(questions, clusters):
#
#     # clusters = {
#     #     "questions": [ 'and the moon too', 'lets show some' ],
#     #     "clusterIds": [ 4, 4 ]
#     # }
#
#     # questions = [ {
#     #         "id": 11, "question": 'Another one about the sun?'
#     #     },
#     #     {
#     #         "id": 33,
#     #         "question": 'What is the distance from the sun though?' },
#     #     {
#     #         "id": 37,
#     #         "question": 'what\'s the changing factors of the sun and moon together?'
#     # } ]
#
#     # Clean and Lemmatize all the questions in the clusters
#     for idx, question in enumerate(clusters["questions"]):
#         question = clean_text(question.replace("\n", ""))
#         clusters["questions"][idx] = " ".join([d.lemma_ for d in nlp(question)])
#
#     # Clean and Lemmatize all the question in the orphan group
#     for idx, question in enumerate(questions):
#         question = clean_text(questions[idx]["question"].replace("\n", ""))
#         questions[idx]["question"] = " ".join([d.lemma_ for d in nlp(question)])
#
#     # Create corpus
#     completeCorpus = clusters["questions"] + [q["question"] for q in questions]
#     # completeCorpus = " ".join(clusters["questions"])
#     # completeCorpus += " ".join([q["question"] for q in questions])
#
#     vectorizer = TfidfVectorizer(
#         analyzer='char',
#         ngram_range=(1, 3),
#         use_idf=True,
#         sublinear_tf=True,
#         smooth_idf=True,
#         stop_words='english'
#     )
#     vectors = vectorizer.fit(completeCorpus)
#
#     # add tfidf vectors to the clusters object for each question
#     clusters["question_vectors"] = vectorizer.transform(clusters["questions"]).toarray()
#     # for idx, question in enumerate(clusters["questions"]):
#     #     clusters["question_vectors"].append(vectorizer.transform(question).toarray())
#
#     #
#     for idx, question in enumerate(questions):
#         question = questions[idx]["question"]
#         questions[idx]["question_vector"] = vectorizer.transform([question]).toarray()[0]
#
#     # print("\n Clusters: ", clusters)
#     # print("\n Questions: ", questions)
#     # print("\n Complete Corpus: ", completeCorpus)
#
#     cluster.findBestFitCluster(questions, clusters)
#
#     return True
#
#
# def dist(wordOne, wordTwo):
#     one = nlp(wordOne)
#     two = nlp(wordTwo)
#     if (np.linalg.norm(one.vector) == 0
#             or np.linalg.norm(two.vector) == 0):
#         raise Exception("Don't have both words")
#     return one.similarity(two)
#
#
# def createCompleteCorpus(questions, clusters):
#     completeCorpus = [q["question"] for q in questions]
#     completeCorpus = completeCorpus + [c["question"] for idx in clusters for c in clusters[idx]]
#     completeCorpus = nlp(" ".join(completeCorpus))
#     completeCorpus = " ".join([d.lemma_ for d in completeCorpus])
#     # print("completeCorpus: ", completeCorpus)
#
#
# def getVector(word):
#     return nlp(word).vector

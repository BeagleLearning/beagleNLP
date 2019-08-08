import math
import document_distance as dd
import numpy as np
from sklearn.naive_bayes import  ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
import math

MAX_DOC_DIST = 1.5


def findKeywordClusters(questions=[], keywords=[]):
    """
    Using the complement naive bayes algorithm find and return the questions that (most *probably*) belong to a given keyword cluster.

    Parameters:
        questions (list(dictionary)): The list of dictionaries containing the question and its Ids
        keywords (list(string)): A list of keyword strings

    Returns:
        clusters (dictionary(list)): Returns a clusters dictionary whose keys are the keywords, and value is an array of all the questionIds that belong to that keyword category

    Note: We assume the questions are the categories and the keyword is what we are trying to categorize - For every keyword we do not calculate the probability of the keyword belonging to that question, rather the probability of the keyword belonging to the other questions, then take the argmin.
    """
    corpus = [q["question"] for q in questions]
    questionIds = [q["id"] for q in questions]

    # Initialize all to empty list to allow for returning empty clusters
    clusters = { kw: [] for kw in keywords }
    mappings = {}

    for kw in keywords:
        ids, threshold = cluster_by_keyword(kw, corpus, questionIds)
        print("Keyword", kw, ids)
        for i in ids:
            if i[0] in mappings.keys():
                # duplicate
                if i[1] > mappings[i[0]][0]:
                    mappings[i[0]] = (i[1], kw)
            else:
                # not duplicate
                mappings[i[0]] = (i[1], kw)

    
    for qnId in mappings:
        keyword = mappings[qnId][1]

        if keyword in clusters.keys():
            clusters[keyword].append(qnId)
        else:
            clusters[keyword] = [qnId]


    uncategorized_questions = [qn for qn in questions if qn["id"] not in mappings.keys()]

    return clusters, uncategorized_questions


# find questions for a keyword
def cluster_by_keyword(keyword=None, questions=[], questionIds=[], analyzer="word"):
    """
    Finds the keyword question pairs using complement naive bayes algorithm by first tfidf vectorizing the list of questions (corpus) and trying to find the questions with the highest likelihood of belonging to the keyword cluster.

    Parameters:
        keyword (string): A string title of the cluster we are trying to look for
        questions (list(string)): The list of the question string text
        questionIds (list(number)): A list containing the ids of the questions passed as the            questions parameter. These Ids are what is returned per identified cluster.
        analyser (string): Either "word", "char", or "char_wb"
            - Use word for identifying whole words and drop potential support for spelling errors, but support stop words and finding exact words.
            - Use char to identify ngrams based on a sequence of characters, does not support stop words and can return less than ideal clusters compositions, but supports spelling mistakes and incomplete keywords
            - Use char_wb to have character level ngrams that do not go beyond word boundaries. Extra padding is added to last characters of words when making ngrams.

    Returns:
        list of cluster tupples (list[(qnId, probability)]): List of tuples containing question Ids 
            and the probability of the question belonging to this cluster
        threshold (number): The dynamic threshold that was used as a cut off point.
    """
    keyword_len = len(keyword)
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.4, smooth_idf=True, stop_words="english",
        ngram_range=(1, 3), analyzer=analyzer, 
        use_idf=True, lowercase=True)
    vectorized_questions = vectorizer.fit_transform(questions)
    
    cnb_clf = ComplementNB(alpha=1, fit_prior=False)
    cnb_clf.fit(vectorized_questions, questionIds)

    
    keyword = vectorizer.transform([keyword])
    probabilities = cnb_clf.predict_proba(keyword)
    # threshold = np.var(probabilities) + np.median(probabilities)
    threshold = np.std(probabilities) + np.median(probabilities) + np.var(probabilities)
    return [(questionIds[ix], p) for ix, p in enumerate(probabilities[0]) if p > threshold], threshold

def clusterOnKeywords(questions, keywords):
    # no keywords - leave everything undefined
    if len(keywords.documents) == 0:
        clusters = {}
        clusters["uncategorized"] = questions.documents
        questions.clusters = clusters
        return questions

    # measure dist from each doc to each keyword
    # for each document, take a tf-weighted distance to each keyword
    # group with the closest keyword. But we have to figure out how to
    # choose if something is too far away. Starting with just an
    # arbitrary line is probably OK
    # we do a one-way distance match for each one
    clusters = dict(zip(
        [k.text for k in keywords.documents],
        [[] for d in keywords.documents]
    ))
    clusters["uncategorized"] = []
    for doc in questions.documents:
        closestKeyword = None
        minDist = math.inf
        for kDoc in keywords.documents:
            if not keywords.atLeastOneTokenKnown(kDoc):
                # skip this one if the document does not have any tokens we
                # know. In that case we can't make an accurate distance measure.
                continue

            dist = dd.directionalWordPairsDistance(kDoc, doc, "cosine")
            if dist < minDist:
                closestKeyword = kDoc
                minDist = dist

        print(f"min dist is {minDist}")
        if minDist < MAX_DOC_DIST:
            clusters[closestKeyword.text].append(doc)
        else:
            clusters["uncategorized"].append(doc)

    questions.clusters = clusters
    return questions

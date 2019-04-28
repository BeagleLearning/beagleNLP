import math
from spacy.tokens import Doc
from wordfreq import word_frequency

from spacy.lang.en import STOP_WORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import regex as re
from collections import OrderedDict

Doc.set_extension("textFrequency", default={})
Doc.set_extension("tfidf", default={})
Doc.set_extension("globalTfidf", default={})


def generateTextFrequency(doc):
    # count the number of occurances of each lemma in the doc
    lemmaList = [token.lemma_ for token in doc]
    tf = {}
    for lemma in lemmaList:
        tf[lemma] = tf.get(lemma, 0) + 1
    # store the full lemma text frequency map on the doc, and return
    doc._.textFrequency = tf
    return doc


def getGlobalIDF(lemma):
    globalDF = word_frequency(lemma, "en")
    if globalDF == 0:
        globalDF = math.pow(1, -9)
    return -math.log(globalDF)


def perDocumentTfidf(corpus):
    def calculateTfidf(doc):
        doc._.tfidf = {}
        doc._.globalTfidf = {}
        for lemma in doc._.textFrequency:
            doc._.tfidf[lemma] = doc._.textFrequency[lemma] * corpus.idf[lemma]
            doc._.globalTfidf[lemma] = doc._.textFrequency[lemma] * \
                getGlobalIDF(lemma)
        return doc

    corpus.documents = [calculateTfidf(d) for d in corpus.documents]
    return corpus


# used during the corpus processing stage
def tfidf(corpus):
    # count how many documents had each lemma at least once
    totalDocCount = len(corpus.documents)
    corpus.idf = {}
    for doc in corpus.documents:
        for lemma in doc._.textFrequency:
            corpus.idf[lemma] = corpus.idf.get(lemma, 1)
    # convert to an idf score
    for lemma in corpus.idf:
        corpus.idf[lemma] = math.log(totalDocCount/corpus.idf[lemma])

    return perDocumentTfidf(corpus)


def tfidfWithExternalData(corpus):
    return corpus


# TODO: Clean
stopwords = STOP_WORDS
def pre_process(text):
    """ Lower case all features and remove tags, numbers and full stops """
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text.lower())
    # TODO: missing check for special characters in OG codebase
    return re.sub("(\\d|\\W)+"," ",text)

def sort_matrix(matrix):
    """ Sort a given matrix in descending order """
    tuples = zip(matrix.col, matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def top_keywords(feature_names, sorted_vectors):
    """ Return the keywords with the highest score value and their score """
    sorted_vectors = sorted_vectors[:5]
    scores, features = [], []
    
    for idx, score in sorted_vectors:
        scores.append(round(score, 3))
        features.append(feature_names[idx])

    results= {}
    for idx in range(len(features)):
        results[features[idx]]=scores[idx]
    
    return results

def get_vocab(sentences):
    """ Creates the vocabulary including the occurancens of the vocab """
    vocab = OrderedDict()
    i = 0
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = i
                i += 1    
    return vocab

tfidf_transformer = TfidfTransformer(norm="l2", smooth_idf=True, use_idf=True, sublinear_tf=False)
def calc_keywords(questions_list, corpus):
    """ Prints out the keywords and their scores given a list of questions """
    vectorizer = CountVectorizer(stop_words=stopwords)
    # word_count_vector = vectorizer.fit_transform(questions_list)
    word_count_vector = vectorizer.fit_transform(corpus)
    
    tfidf_transformer.fit(word_count_vector)
    features = vectorizer.get_feature_names()
    
    doc = ", ".join(questions_list)
    tfidf_vector = tfidf_transformer.transform(vectorizer.transform([doc]))
    
    sorted_vectors = sort_matrix(tfidf_vector.tocoo())
    keywords = top_keywords(features, sorted_vectors)
    
    return max(keywords, key=keywords.get)
    # print(doc)
    # print("\nKeywords")
    # for k in keywords:
    #     print(k,keywords[k])
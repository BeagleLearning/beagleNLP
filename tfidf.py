import math
from spacy.tokens import Doc
Doc.set_extension("textFrequency", default={})
Doc.set_extension("tfidf", default={})


# used during the pipeline of document processing to generate frequencies
def generateTextFrequency(doc):
    # count the number of occurances of each lemma in the doc
    lemmaList = [token.lemma_ for token in doc]
    tf = {}
    for lemma in lemmaList:
        tf[lemma] = tf.get(lemma, 0) + 1
    # store the full lemma text frequency map on the doc, and return
    doc._.textFrequency = tf
    return doc


def perDocumentTfidf(corpus):
    def calculateTfidf(doc):
        doc._.tfidf = {}
        for lemma in doc._.textFrequency:
            doc._.tfidf[lemma] = doc._.textFrequency[lemma] * corpus.idf[lemma]
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

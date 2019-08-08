"""
TextRank implementation for tagging clusters with keyword(s)
"""
import regex as re
from collections import OrderedDict
import numpy as np
import analysis
import operator
import time

parts_of_speech = ["NOUN", "PROPN", "VERB"]
dampen_const = 0.85
min_diff = 1e-5
# Potentially dynamic and set to be relative to the average length of sentences(???)
window_size = 2

# Note: possibly ignore POS when the corpus size is small(???)
def process_corpus(corpus, parts_of_speech=parts_of_speech, ignore_POS=False):
    """
    Process corpus to remove stopwords and return lemmatized words
    
    Parameters:
        corpus (list): List of spacy documents
        parts_of_speech (list): List of they types of words we want to pay attention to
            (default is defined above as parts_of_speech)
        ignore_POS (bool): Whether or not to ignore the POS and consider every word
    
    Returns:
        sentences (list): List of processed documents that only contain the parts of speech needed
    """
    sentences = []
    for document in corpus:
        for sent in document.sents:
            words = []
            for token in sent:
                if ignore_POS:
                    if token.is_stop is False and token.is_punct is False:
                        words.append(token.lemma_)
                else:
                    if token.pos_ in parts_of_speech and token.is_stop is False and token.is_punct is False:
                        words.append(token.lemma_)
            sentences.append(words)
    return sentences



def get_vocab(sentences):
    """ 
    Calculates the text frequency of a word in a sentence

    Parameters:
        sentences (list): A list of strings (lemma_) that make up a given cluster

    Returns:
        vocabulary (OrderedDict):  collection of keywords and their text frequency
    """

    vocab = OrderedDict()
    i = 0
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = i
                i += 1
    return vocab



def token_pairs(sentences):
    """ 
    Get the words that are linking to each other

    Parameters:
        sentences (list): List of spacy documents

    Returns:
        pairs (list): List of tuples that contain two words that link to one another (word1, word2)
    """

    pairs = list()
    for sentence in sentences:
        for ix, word in enumerate(sentence):
            for jx in range(ix+1, ix+window_size):
                if jx >= len(sentence):
                    break
                pair = (word, sentence[jx])
                if pair not in pairs:
                    pairs.append(pair)
    # print("Token pairs")
    # print(pairs)
    return pairs



def get_matrix(vocab, token_pairs):
    """ 
    Returns normalized matrix representing the inboud and outboud links of each word

    Parameters:
        vocab (OrderedDict): collection of words and their frequency

    Returns:
        normalized matrix (list): A normalized matrix showing the weight (textrank) of each word
    """

    # Build matrix
    vocab_size = len(vocab)
    g = np.zeros((vocab_size, vocab_size), dtype='float')
    for word1, word2 in token_pairs:
        i, j = vocab[word1], vocab[word2]
        g[i][j] = 1

    g = g + g.T - np.diag(g.diagonal())

    # Normalize matrix by column
    norm = np.sum(g, axis=0)
    g_norm = np.divide(g, norm, where=norm != 0)

    return g_norm


def remove_stopwords(corpus):
    """ Returs list of tokens excluding stop words """
    return [token for token in corpus.doc if not token.is_stop]


def get_keywords(cluster, corpus, return_one=True, use_idf=False):
    """
    Identifies the keywords in a list of questions using the TextRank algorithm

    Parameters:
        cluster (list): A list of spacy Doc strings that make up a given cluster
        return_one (bool): Flag to determine whether we return only one keyword or not
                (default is True)
    Returns:
        keyword (str): a single keyword if flag `return_one` is set to true,
        node_weights (OrderedDict):  collection of keywords and their textrank score
    """

    sentences = process_corpus(cluster)
    vocab = get_vocab(sentences)
    tokens = token_pairs(sentences)
    matrix = get_matrix(vocab, tokens)

    init_rank = np.array([1] * len(vocab))
    previous_rank = 0
    
    for epoch in range(10):
        init_rank = (1 - dampen_const) + dampen_const * \
            np.dot(matrix, init_rank)
        if abs(previous_rank - sum(init_rank)) < min_diff:
            break
        else:
            previous_rank = sum(init_rank)

    start = time.time()
    node_weight = OrderedDict()
    for word, index in vocab.items():
        node_weight[word] = init_rank[index]

    node_weight = OrderedDict(
        sorted(node_weight.items(), key=lambda item: item[1]))


    # print("\n Node Weights per cluster: ")
    # print(cluster)
    # print(node_weight)
    # print("\n")

    kwords = list(node_weight)
    keyword = "uncategorized"
    if return_one:
        if use_idf:
            ranks = list(node_weight.values())
            trIDFScores = {}
            for ix, word in enumerate(kwords):
                idfScore = 0.1
                try:
                    idfScore = corpus.idf_[corpus.vocabulary_[word]]
                except:
                    pass
                trIDFScores[word] = ranks[ix] * (idfScore / len(kwords))

                try:
                    max_sorted = max(trIDFScores.items(), key=operator.itemgetter(1))
                    keyword = max_sorted[0]
                except:
                    pass
                return keyword
        else:
            try:
                keyword = list(node_weight)[-1]
            except:
                pass
            return keyword

    # if return_one:
    #     k_word = "uncategorized"
    #     try:
    #         max_sorted = max(trIDFScores.items(), key=operator.itemgetter(1))
    #         k_word = max_sorted[0] #+ " & " + max_sorted[1]
    #         # k_word = list(node_weight)[-1]
    #     except:
    #         pass
    #     return k_word

    return node_weight

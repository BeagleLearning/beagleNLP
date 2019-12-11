"""
TextRank implementation for tagging clusters with keyword(s)
"""
import regex as re
from collections import OrderedDict
import spacy as sp
import numpy as np

parts_of_speech = ["NOUN", "PROPN", "VERB"]
dampen_const = 0.85
min_diff = 1e-5
window_size = 3 # Potentially dynamic and set to be relative to the average length of sentences(???)

def clean_text(text, lower=False):
    """ Lower case all features and remove tags, numbers and full stops """
    text = text.lower() if lower else text
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
    return re.sub("(\\d|\\W)+"," ",text)

# Note: possibly ignore POS when the corpus size is small(???)
def process_corpus(corpus, parts_of_speech=parts_of_speech, ignore_POS=False):
    """
    Takes a document and:
    1. filters out stop words
    2. only returns words with one of the defined part of speech
    """
    sentences = []
    for sent in corpus.sents:
        words = []
        for token in sent:
            if ignore_POS:
                if token.is_stop is False and token.is_punct is False:
                    words.append(token.text)
            else:
                if token.pos_ in parts_of_speech and token.is_stop is False and token.is_punct is False:
                    words.append(token.text)
        sentences.append(words)
    return sentences


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


def token_pairs(sentences):
    """ Returns the pairs of tokens next to each other """
    pairs = list()
    for sentence in sentences:
        for ix, word in enumerate(sentence):
            for jx in range(ix+1, ix+window_size):
                if jx >= len(sentence):
                    break
                pair = (word, sentence[jx])
                if pair not in pairs:
                    pairs.append(pair)
    return pairs


def get_matrix(vocab, token_pairs):
        """ Returns normalized matrix """
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        g = g + g.T - np.diag(g.diagonal())
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0)
        
        return g_norm


def remove_stopwords(corpus):
    """ Returs list of tokens excluding stop words """
    return [token for token in corpus.doc if not token.is_stop]


def get_keywords(corpus, return_one=True):
    """ 
    Takes a list of questions in a cluster, and whether to return just one keyword or a list of them
    Returns the keyword recommendation(s)
    """
    sentences = process_corpus(corpus)
    vocab = get_vocab(sentences)
    tokens = token_pairs(sentences)
    matrix = get_matrix(vocab, token_pairs(sentences))
    
    init_rank = np.array([1] * len(vocab))
    previous_rank = 0
    # random walk(??)
    for epoch in range(10):
        init_rank = (1 - dampen_const) + dampen_const * np.dot(matrix, init_rank)
        if abs(previous_rank - sum(init_rank))  < min_diff:
            break
        else:
            previous_rank = sum(init_rank)
            
            
    node_weight = OrderedDict()
    for word, index in vocab.items():
        node_weight[word] = init_rank[index]

    node_weight = OrderedDict(sorted(node_weight.items(), key=lambda item: item[1]))

    # print("\n")
    # print(node_weight)
    # print("\n")
    if return_one:
        k_word = "uncategorized"
        try:
            k_word = list(node_weight)[-1]
        except:
            pass
        return k_word

    return node_weight
"""Produces keyword-based clusters from a corpus using a greedy algorithm"""
from math import ceil
from spacy.lang.en import STOP_WORDS
from wordfreq import word_frequency

MIN_CLUSTER_FRACTION = 0.08
MAX_CLUSTER_FRACTION = 0.2
MIN_CLUSTER_ABSOLUTE = 2

def __add_keyword_clusters(corpus):
    """Constructs a potential cluster choice for every keyword in the set of docs.

    Args:
        corpus (TaggedQuestionCorpus)
    """
    corpus.all_possible_keyword_clusters = {}
    for doc in corpus.documents:
        for lemma in doc._.textFrequency:
            kcs = corpus.all_possible_keyword_clusters
            if not lemma in kcs:
                kcs[lemma] = set()
            kcs[lemma].add(doc)

def __find_next_cluster(keyword_clusters, min_cluster_size, max_cluster_size):
    """Chooses the best available cluster larger than min_cluster_size and smaller than
    max_cluster_size, where "best" means having the keyword that is rarest in the English language
    (according to wordfreq's corpus) among the options options.

    Args:
        keyword_clusters (dict): set of documents for each keyword
        min_cluster_size (int)
        max_cluster_size (int)
    """
    lemma_dfs = [(k, len(v)) for k, v in keyword_clusters.items()]
    lemma_dfs.sort(key=lambda a: a[1])
    possible_keywords = set()
    for lemma_df in lemma_dfs:
        if lemma_df[1] < min_cluster_size or lemma_df[0] in STOP_WORDS or lemma_df[0] == "-PRON-":
            continue
        if lemma_df[1] > max_cluster_size:
            break
        possible_keywords.add(lemma_df[0])
    if len(possible_keywords) == 0:
        return None, None
    rarest_freq = 1
    rarest_keyword = None
    for keyword in possible_keywords:
        freq = word_frequency(keyword, "en")
        if 0.0 < freq < rarest_freq:
            rarest_keyword = keyword
            rarest_freq = freq
    return rarest_keyword, keyword_clusters[rarest_keyword]

def cluster_greedily(corpus):
    """Generates keyword-based clusters from a corpus using a greedy algorithm.
    Args:
        corpus (TaggedQuestionCorpus)
    """
    __add_keyword_clusters(corpus)
    kcs = corpus.all_possible_keyword_clusters.copy()
    min_cluster_size = int(ceil(len(corpus.documents) * MIN_CLUSTER_FRACTION))
    min_cluster_size = max(min_cluster_size, MIN_CLUSTER_ABSOLUTE)
    max_cluster_size = int(ceil(len(corpus.documents) * MAX_CLUSTER_FRACTION))
    clusters = {}
    next_keyword, next_cluster = __find_next_cluster(kcs, min_cluster_size, max_cluster_size)
    uncategorized_docs = set(corpus.documents)
    while next_cluster:
        clusters[next_keyword] = next_cluster
        del kcs[next_keyword]
        uncategorized_docs = uncategorized_docs.difference(next_cluster)
        for doc_set in kcs.values():
            for doc in next_cluster:
                doc_set.discard(doc)
        next_keyword, next_cluster = __find_next_cluster(kcs, min_cluster_size, max_cluster_size)
    clusters["uncategorized"] = uncategorized_docs
    corpus.clusters = clusters
    return corpus

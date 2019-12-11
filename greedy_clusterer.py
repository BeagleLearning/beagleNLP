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
    corpus.keyword_clusters = {}
    for doc in corpus.documents:
        for lemma in doc._.textFrequency:
            kcs = corpus.keyword_clusters
            if not lemma in kcs:
                kcs[lemma] = set()
            kcs[lemma].add(doc)

def __find_next_cluster(keyword_clusters, min_cluster_size, max_cluster_size):
    """Chooses the smallest available cluster greater than min_cluster_size to include
    in final set of clusters returned by cluster_greedily.

    Args:
        keyword_clusters (dict)
        min_cluster_size (int)
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
        return None
    rarest_freq = 1
    rarest_keyword = None
    print("Possible keywords:", possible_keywords)
    for keyword in possible_keywords:
        freq = word_frequency(keyword, "en")
        if 0.0 < freq < rarest_freq:
            rarest_keyword = keyword
            rarest_freq = freq
    print("Chosen:", rarest_keyword, word_frequency(rarest_keyword, "en"))
    return rarest_keyword, keyword_clusters[rarest_keyword]

def cluster_greedily(corpus):
    """Generates keyword-based clusters from a corpus using a greedy algorithm.
    Args:
        corpus (TaggedQuestionCorpus)
    """
    __add_keyword_clusters(corpus)
    kcs = corpus.keyword_clusters.copy()
    min_cluster_size = int(ceil(len(corpus.documents) * MIN_CLUSTER_FRACTION))
    min_cluster_size = max(min_cluster_size, MIN_CLUSTER_ABSOLUTE)
    max_cluster_size = int(ceil(len(corpus.documents) * MAX_CLUSTER_FRACTION))
    clusters = []
    next_cluster = __find_next_cluster(kcs, min_cluster_size, max_cluster_size)
    uncategorized_docs = set(corpus.documents)
    while next_cluster:
        clusters.append(next_cluster)
        used_docs = next_cluster[1]
        del kcs[next_cluster[0]]
        uncategorized_docs = uncategorized_docs.difference(used_docs)
        for doc_set in kcs.values():
            for doc in used_docs:
                doc_set.discard(doc)
        next_cluster = __find_next_cluster(kcs, min_cluster_size, max_cluster_size)
    clusters.append(("uncategorized", uncategorized_docs))
    corpus.clusters = {c[0]: c[1] for c in clusters}
    return corpus

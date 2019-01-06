def buildTagCluster(corpus):
    return {k: [doc._.tag for doc in v] for (k, v) in corpus.clusters.items()}

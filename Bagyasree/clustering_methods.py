from sklearn.cluster import AgglomerativeClustering, KMeans

class ClusteringMethods:
    def agglomerative_clustering(data, no_of_clusters): 
        cluster = AgglomerativeClustering(n_clusters = no_of_clusters, affinity = 'euclidean', linkage = 'ward')
        clusters = cluster.fit_predict(data)
        return clusters
    
    #Need to check for errors
    def kmeans_clustering(data, no_of_clusters):
        kmeans_cluster = KMeans(n_clusters = no_of_clusters)
        clusters = kmeans_cluster.fit_predict(data)
        return clusters
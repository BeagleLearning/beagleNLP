from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
from helper_functions import intracluster_similarity, intercluster_similarity

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

    def gaussian_mixture_models(data, no_of_clusters):
        #no_of_clusters = 25
        GMM = GaussianMixture(n_components=no_of_clusters, covariance_type='full', n_init = 5)
        GMM.fit(data)
        cluster_probabilities = GMM.predict_proba(data)
        # print(cluster_probabilities)
        #print(GMM.eval())
        # epsilon = 0.005    
        # print(GMM.predict_proba([GMM.means_[0]+epsilon]))
        clusters = []
        for prob in cluster_probabilities:
            clusters.append(np.argmax(prob))
        #clusters = GMM.predict(data)

        print("INTRACLUSTER: ")
        print()
        similarities = intracluster_similarity(clusters,data)
        number = 1
        for cluster in similarities:
            print("Cluster ",number, ": ",cluster)
            number +=1
        print()
        print("INTERCLUSTER: ")
        print()
        similarities = intercluster_similarity(clusters,data)
        for cluster_pair in similarities:
            print("Clusters ",cluster_pair, ": ",similarities[cluster_pair])
        print()

        return clusters
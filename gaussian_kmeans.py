import numpy as np
from scipy.stats import anderson
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import scale
import warnings
import math

min_obs = 2  # absolute minimum per cluster!
max_depth = 3  # SHOULD be dynamic based on the number of questions submitted
random_state = 100
strictness = 0 # SHOULD be dynamic based on the number of questions submitted 0 <= strictness < 5

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Cluster:
    """
    Mimmic the sklearn cluster class for ease of use with other methods down the line.

    Attributes:
        labels_ (list): List of the cluster item labels
        criteria_ (list): List of the stopping criteria per cluster
    """
    
    def __init__(self, labels_ = [], criteria_=[]):
        self.labels_ = labels_
        self.criteria_ = criteria_

    def labels(self):
        return self.labels_


def gaussian_check(vector, strictness=strictness):
    """
    Statistical anderson-darling test to check if a given vector is gaussian

    Args:
        vector (list): A normalized and scaled vector difference between two (kmeans identified) clusters
        strictness (int): Tolerance value, for just how much variation can be considered gaussian.

    Returns:
        A boolean value representing whether the vector is gaussian or not
    """
    output = anderson(vector)
    #print(output[1],output[1][strictness])
    if output[0] <= output[1][strictness]:
        return True
    else:
        return False


def clustering(data, depth, data_index, index, stopping_criteria, max_depth, min_obs, strictness):
    """
    Clustering done by recursively run kmeans with k=2 on your data until a max_depth is reached or we have gaussian clusters

    Args:
        data (nd array): A representation matrix of the documents to be clustered
        depth (int): Integer used to determine the current level of depth in the recursion
        data_index (list): A list keeping track of the data points and their categories
        index (list): A list containing the current indexes per cluster
        stopping_criteria (list): List keeping track of how many times we stopped the recursion and why - one of: min_obs, max_depth, or gaussian.
        max_depth (int): Limit of the depth to which we can recursively split our data.
        min_obs (int): Minimum number of elements in a cluster
        strictness (int): Tolerance value, for just how much variation can be considered gaussian.

    Returns:
        Function either terminates at end of recursion or calls itself.
    """

    depth += 1
    if depth == max_depth:
        data_index[index[:, 0]] = index
        stopping_criteria.append('max_depth')
        return

    km = KMeans(n_clusters=2, random_state=random_state)
    # km = MiniBatchKMeans(n_clusters=2, random_state=random_state)
    km.fit(data)

    centers = km.cluster_centers_
    v = centers[0] - centers[1]

    
    x_prime = scale(data.dot(v) / (v.dot(v)))
    gaussian = gaussian_check(x_prime)

    if gaussian == True:
        data_index[index[:, 0]] = index
        stopping_criteria.append('gaussian')
        return

    labels = set(km.labels_)
    for k in labels:
        current_data = data[km.labels_ == k]
        if current_data.shape[0] <= min_obs:
            data_index[index[:, 0]] = index
            stopping_criteria.append('min_obs')
            return

        current_index = index[km.labels_ == k]
        current_index[:, 1] = depth
        clustering(current_data, depth, data_index, current_index, stopping_criteria, max_depth, min_obs, strictness)



def get_clusters(data, strictness=strictness, outliers=[]):
    """
    Uses Gaussian K Means to recursively identify clusters whose elements seem to come from the same gaussian distribution.

    Args:
        data (nd array): A representation matrix of the documents to be clustered
        strictness (int): An integer value used to represent the tolerance of the anderson-darling test.
        outliers (list): Optional list of questions to ignore during clustering, and return them as uncategoriesed.

    Returns:
        cluster: A cluster instance with labels and stopping criteria (for debugging, or possibly deciding next steps if we had to stop because of anything other than gaussian results)
    """

    stopping_criteria = []
    min_obs= 1
    max_depth = 200
    index = np.array([(i, False) for i in range(data.shape[0])])
    data_index = np.array(index, copy=True)
    clustering(data, 0, data_index, index, stopping_criteria, max_depth, min_obs, strictness)
    mapping = [i[1] - i[0] for i in enumerate(outliers)]
    labels = np.insert(data_index[:, 1], mapping, -1)
    cluster = Cluster(labels, stopping_criteria)
    # print(cluster.labels_)
    return cluster

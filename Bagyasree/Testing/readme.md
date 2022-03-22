STEPS TO TEST A CLUSTERING ALGORITHM:
1.  Import class Test into your file. 
    from testing import Test
2.  Initialize an object of class test. 
    t = Test()
3.  Use method test_clustering_algorithm with display_clusters set to True if you would like to view the clusters (it is False by default). The parameter 'arguments' is optional
    t.test_clustering_algorithm(algorithm, arguments, display_clusters = True) 


INPUT AND OUTPUT:
The clustering algorithm must take in a list of questions and return a list of cluster numbers (integers), with the number at index i corresponding to question i in the input list. If the algorithm requires any additional positional arguments, these can be passed as a list to the parameter 'arguments' of the test_clustering_algorithm method.


EVALUATION METRICS:
The test class will return three sets of scores - the Adjusted Rand Index scores, the Fowlkes-Mallows scores, and the Normalised Mutual Information scores. 
RI Scores:  Range from -1 to 1. Negative values or values close to 0 are bad; these indicate a high degree of mismatch or randomness in the algorithm's assignments. The higher the value, the better the clusters. A score of 1.0 indicates a perfect match.
FM Scores:  Range from 0 to 1. The higher the score, the better the algorithm.
NMI Scores: Range from 0 to 1. The higher the score, the better the algorithm.
(   For brief explanations of the metrics, visit the following -
    https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index ).


INTERPRETING THE RESULTS:
The test class evaluates the algorithm across 5 different cluster sets, as present in the test data. Each of these sets is a different type of clustering, i.e., questions are grouped differently. One may be a more broad grouping, while another may be more specific. The results for each cluster set are presented in the columns. Each row corresponds to one assignment. Clustering results are scored independently for each assignment. Thus, if an algorithm returns high scores for most assignments down one cluster set, then it can be inferred that the algorithm consistently performs clustering similar to that cluster set.


TEST DATA: 
The test data file path is specified in testing.py, line 10. This must be changed to reflect the path in the working system. Addition or removal of assignments or cluster sets will not affect the test class (If assignments are to be added, please ensure that the assignement IDs are added correctly in the right column. If a different type of clustering is to be tested, please create a new column for the cluster set, with the column name in the same format as the others).

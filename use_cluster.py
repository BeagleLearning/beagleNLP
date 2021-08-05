"""
This is Araz's code to cluster questions from a particular class.
This has 2 key aspects:
1. Using Universal Sentence Encoder for encoding the questions
2. Using 2 different types of Hierarchical Agglomerative Clustering

Read about USE here:https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15
Read about HAC here: https://en.wikipedia.org/wiki/Hierarchical_clustering

The code is well documented, and divided into 4 main functions, and other sub-functions called by them.

For more details on this approach, you can contact Araz at: arazsharma1103@gmail.com
"""

#Importing required Libraries

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import re
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn import metrics
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn import decomposition
import json
from beagleError import BeagleError
import errors
import logging

#Loading the USE Model with Tensorflow Hub

try:
    # # os.environ['TFHUB_CACHE_DIR'] = "C:/Users/arazs/Documents/GitHub/beagleNLP/test_hub"
    
    # embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    logging.debug('Model Loaded')
except:
    raise BeagleError(errors.USE_LOAD_ERROR) #If model not loaded


#Global Variables

#Declaring Range of Clusters to try from
MIN_CLUSTER = 4
MAX_CLUSTER = 11


"""
> The objective of this function is to convert the data recieved by the server into embeddings with the USE Model

:The data is recieved in form of a list of dictionaries with keys: Question, Question_Id, Course_Id
:The questions are stored seperately, and embedded with USE
:The function handles errors for missing/wrong keys & invalid datatypes for the entries
"""

def get_data_embeddings(data):
    questions_list = [] #To store all questions in a list
    question_ids_list = [] #To store all q_ids
    for i in data:
        if('question' not in i or 'question_id' not in i or 'course_id' not in i):
            raise BeagleError(errors.KEY_ERROR)
        if(type(i['question']) != str or type(i['question_id']) != int or type(i['course_id']) not in [int, str]):
            raise BeagleError(errors.INVALID_DATA_TYPE)
        questions_list.append(i['question'])
        question_ids_list.append(i['question_id'])
    logging.debug('Questions extracted Successfully')
    try:
        embeddings = embed(questions_list) #USE embeds data
    except:
        raise BeagleError(errors.USE_EMBED_ERROR)
    logging.debug('Embedded Successfully')
    #logging.debug(embeddings)
    logging.debug(questions_list)
    #logging.debug(question_ids_list)
    return (embeddings, questions_list, question_ids_list) #returns data embeddings and questions to cluster in a list


"""
> The objective of this function is to find the row & column in matrix
given the size of matrix & the position in upper half of the matrix (not considering diagnal)

For example consider a 3x3 matrix

1 2 3
4 5 6
7 8 9

Upper half has numbers 2 3 6 @ positions 1 2 3
So for 6 we want to output: row, column = 2,3 (which would be 1,2 as indexing starts from 0)

The logic for finding this is as follows:
1. The upper half of a matrix consists of numbers in each row as positions-

1 2.....(n-1)   n-1 numbers
  n.........    n-2 numbers
  (2n-2)....    n-3 numbers
     ........
       ......
         ....
        n(n-1)/2  1 number

2. For the position given, we can keep subtracting n-1, n-2... till the value becomes <=0
3. At that point we will know the row this position belongs to
4. For column, we can add the row number to the previous value of position before it became <=0
5. While returning account for index starting @ 0
"""

def get_ind(n, nmb): #n is size of matrix (nxn), nmb is the number at the top half of matrix for which we want row,col indexes
      val = nmb
      strt = n-1
      cnt = 0 #keep count before value becomes <=0
      prev = nmb #keep previous to find column
      while val>0:
        prev = val
        val = val - strt
        strt-=1
        cnt+=1
        
      #now val is <=0
      #cnt-1 is row number index
      #cnt + prev - 1 is column number index
      #logging.debug(cnt)
      #logging.debug(strt)
      #logging.debug(prev)
      return (cnt-1, (cnt + prev - 1)) #returns row, column




"""
> The objective of this function is to get clusters with input as a similarity matrix


:The inputs are number of clusters to return, the questions to cluster & similarity matrix
:The HAC is done with Scipy library
"""


def get_clusters_from_sim_matrix(n_clus, questions_list, X):
    Z = linkage(X,'ward') #Hierarchical Agglomerative Clustering for the input Similarity Matrix
    q_clusters = cut_tree(Z, n_clusters= n_clus).flatten() #Getting Question Clusters for n_clus number of clusters

    grouped = {} #Dictionary to store final clusters
    for i in range(n_clus):
        grouped[i] = []

    
    for i in range(len(q_clusters)):
        grouped[q_clusters[i]].append(questions_list[i])    

    cluster_lens = [len(grouped[i]) for i in grouped.keys()]

    logging.debug('Cluster Sizes are:', cluster_lens)

    logging.debug(q_clusters)
    logging.debug(grouped)
    return (grouped, q_clusters)
  

"""
> The objective of this function is to cluster questions using Hierarichal Agglomerative Clustering 
with Sparsification. 

:HAC with sparsification is a novel approach, which modifies the input similarity matrix before feeding it to the HAC Algorithm
:To read about this approach in detail, see - 
https://www.researchgate.net/publication/328034827_Improving_Short_Text_Clustering_by_Similarity_Matrix_Sparsification

:The inputs for the function are number of clusters to form, data embeddings, list of questions & type of sparsification to use (1 or 2)
:The High level function overview is as follows-
1. Create a similarity matrix from the data embeddings
2. Modify the matrix by aij = (sij - ui)/std.dev(i) 
3. From this matrix, extract all max values of complements
4. Sort this list
5. Define threshold as: [nxl/2] for l as: n/C  (n: Total Number of Qs, C: No. of Clusters)
6. Type 1 uses threshold & Type 2 uses 2*threshold
7. From list keep top values according to required threshold
8. All left values are ones to be sparsed, i.e made to be 0
9. Get indexes of left values in original matrix & make them 0
10.Feed this final matrix into HAC algorithm to create clusters
"""


def HAC_with_Sparsification(n_clus, embeddings, questions_list, type_n = 1, return_mat = 0):
  number_of_qs = embeddings.shape[0] #Total number of questions

  #creating initial similarity matrix from data embeddings

  initial_sim_matrix = metrics.pairwise.cosine_similarity(embeddings)
  mean_list = [] #list to store means of rows
  std_list = [] #list to store std deviation of rows

  for i in range(len(initial_sim_matrix)):
    mean_list.append(np.mean(np.concatenate((initial_sim_matrix[i][:i],initial_sim_matrix[i][i+1:]))))
    std_list.append(np.std(np.concatenate((initial_sim_matrix[i][:i],initial_sim_matrix[i][i+1:]))))

  modified_sim = initial_sim_matrix.copy()

  #creating the modified similarity matrix

  for i in range(len(initial_sim_matrix)):
    modified_sim[i] = (initial_sim_matrix[i] - mean_list[i])/std_list[i]

  #list to store max of values of all complements
  L = []
  for i in range(len(modified_sim)):
    for j in range(i+1, len(modified_sim)):
      #j>i guaranteed
      L.append((max(modified_sim[i][j],modified_sim[j][i]),initial_sim_matrix[i][j]))

  #Now to sort the list
  L_mod = []
  for i,j in enumerate(L): #enumerating to have indexes in the list
    L_mod.append((i+1,j))

  L_mod.sort(key = lambda x: x[1][0], reverse = True)

  #Defining the Threshold

  #threshold = int((number_of_qs*((number_of_qs/n_clus)-1))/2) Can have n/C - 1 too
  threshold = int((number_of_qs*((number_of_qs/n_clus)))/2)

  #Sparsifying the Matrix 

  if(type_n==2):
    #Creating the type 2 (2M threshold) Similarity Matrix
    thresholded_list2 = sorted(L_mod[(2*threshold):],key = lambda x: x[1][1], reverse=True)

    M2_sim_matrix = initial_sim_matrix.copy() #Final Type 2 Similarity Matrix


    for i in thresholded_list2:
      row, col = get_ind(number_of_qs, i[0])
      M2_sim_matrix[row][col] = 0.0
      M2_sim_matrix[col][row] = 0.0

    X = 1 - M2_sim_matrix
  else:

    #Creating the type 1 (M threshold) Similarity Matrix
    thresholded_list = sorted(L_mod[threshold:],key = lambda x: x[1][1], reverse=True)

    M_sim_matrix = initial_sim_matrix.copy() #Final Type 1 Similarity Matrix

    for i in thresholded_list:
  
        row, col = get_ind(number_of_qs, i[0])
        
        M_sim_matrix[row][col] = 0.0
        M_sim_matrix[col][row] = 0.0

    X = 1 - M_sim_matrix

  #For seeing the sparsified matrix
  logging.debug(X)

  grouped, q_clusters = get_clusters_from_sim_matrix(n_clus, questions_list, X)

  logging.debug(grouped)
  logging.debug(q_clusters)
  if(return_mat==0):
    return (grouped, q_clusters)
  else:
      return (grouped, q_clusters, X)



"""
> The objective of this function is to evaluate the best number of clusters & call the HAC with sparsification
function to return those clusters

:The inputs required are the embeddings, questions to cluster & type of sparsification (1 or 2)
:Type 1 uses the normal threshold
:Type 2 uses 2*normal threshold
:The number of clusters tested are from 4 till 10
:Silhouette Score is used to decide the best cluster number
:That cluster number is used to output the best clusters stored
:The function handles errors for number of questions less than cluster number
"""



def best_score_HAC_sparse(data_embeddings, questions_list, type_n = 1):
    n_clus_options = range(MIN_CLUSTER,MAX_CLUSTER) #number of clusters from 4 to 10
    scores = [] #stores scores for each cluster number
    q_clus_list = [] #stores q_cluster list (for assigning questions to cluster) for each cluster number
    grp_list = [] #stores grouped list(final clusters) for each cluster number
    for k in n_clus_options:
        if(k < len(questions_list)):
            try:
                grouped, q_clusters = HAC_with_Sparsification(k, data_embeddings, questions_list,type_n)
            except:
                raise BeagleError(errors.HAC_SPARS_FUN_ERROR) #Error for Number of qs <= Cluster number
            q_clus_list.append(q_clusters)
            grp_list.append(grouped)
            scores.append([k, metrics.silhouette_score(data_embeddings, q_clusters, metric='euclidean')])

    logging.debug(scores)
    best_cluster_number = sorted(scores, key = lambda x: x[1], reverse = True)[0][0]
    logging.debug("Optimal Cluster Number with silhoutte score is:",best_cluster_number)
    
    return (grp_list[best_cluster_number- MIN_CLUSTER],q_clus_list[best_cluster_number- MIN_CLUSTER]) #returns q_cluster list & final clusters for best cluster number

"""
> The objective of this function is to evaluate the best number of clusters & call the Standard HAC function to return those clusters

:The inputs required are the embeddings & questions to cluster
:Uses Agglomerative Clustering from sklearn
:The number of clusters tested are from 4 till 10
:Silhouette Score is used to decide the best cluster number
:That cluster number is used to output the best clusters stored
:The function handles errors for number of questions less than cluster number
"""

#NORMAL HAC
def get_best_HAC_normal(data_embeddings, questions_list):
        n_clus_options = range(MIN_CLUSTER,MAX_CLUSTER) #number of clusters from 4 to 10
        scores = [] #stores scores for each cluster number
        q_clust_list = [] #stores q_cluster list (for assigning questions to cluster) for each cluster number

        for k in n_clus_options:
            if(k < len(questions_list)):
                try:
                    cluster = AgglomerativeClustering(k, affinity = 'euclidean', linkage = 'ward')
                    q_clusters = cluster.fit_predict(data_embeddings)
                except:
                    raise BeagleError(errors.HAC_FUN_ERROR) #Error for Number of qs <= Cluster number
                q_clust_list.append(q_clusters)
                scores.append([k, metrics.silhouette_score(data_embeddings, q_clusters, metric='euclidean')])

        logging.debug(scores)
        best_cluster_number = sorted(scores, key = lambda x: x[1], reverse = True)[0][0]
        logging.debug("Optimal Cluster Number with silhoutte score is:",best_cluster_number)
        q_cluster_final = q_clust_list[best_cluster_number - MIN_CLUSTER] #Final q_cluster list for best evaluated cluster

        
        grouped = {} #Stores the final clusters
        for i in range(best_cluster_number):
            grouped[i] = []

        
        for i in range(len(q_cluster_final)):
            grouped[q_cluster_final[i]].append(questions_list[i])    

        cluster_lens = [len(grouped[i]) for i in grouped.keys()]

        logging.debug('Cluster Sizes are:', cluster_lens)
        logging.debug(grouped)
        logging.debug(q_cluster_final)
        return (grouped, q_cluster_final)


"""
> The objective of this function is to return clusters in format of {cluster_id: [q_ids in cluster]}

:The inputs required are the q_cluster list & question id's list
:The dictionary is formed with q_ids fed into each cluster number it is alloted
"""

#Returning Cluster_ID: [Q_Ids]
def return_cluster_dict(q_cluster, q_ids_list):
    final_clusters = {}
    for indx in range(len(q_cluster)):
        if q_cluster[indx] not in final_clusters:
            final_clusters[q_cluster[indx]] = [q_ids_list[indx]]
        else:
            final_clusters[q_cluster[indx]].append(q_ids_list[indx])

    return final_clusters
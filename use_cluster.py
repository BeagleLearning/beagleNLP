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

from numpy.testing._private.utils import assert_equal
from sklearn import cluster
import spacy
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import math
import os
import re
from scipy.cluster.hierarchy import centroid, dendrogram, linkage, cut_tree
from labelling_cluster_corpus import LabellingClusterCorpus
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
    #os.environ['TFHUB_CACHE_DIR'] = "C:/Users/arazs/Documents/GitHub/beagleNLP/test_hub"
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    nlp = spacy.load("en_core_web_lg")
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


"""
> The objective of this function is to generate labels for each cluster based on a custom score developed by Araz, 
using Normalised Mutual Information (NMI) and Distance from the Cluster Centroid 
:The inputs required are the embeddings (generated previously from cluster functions), list of questions, question ids & clusters generated 
: Read about NMI here: https://arxiv.org/pdf/1702.08199.pdf
: For how Araz developed the custom score/metric, refer here: 
"""

def return_cluster_labels_nmi_ngrams_centroid(embeddings,qs_list,q_ids_list,clusters): 

    labelling_corpus = LabellingClusterCorpus(qs_list, nlp, embeddings, q_ids_list, clusters)
    
    cluster_objs, lemmatized_qs_list = generate_modified_qs_data(labelling_corpus.documents, clusters, q_ids_list,1)
    global_keywords = generate_global_keywords(labelling_corpus.documents, 1)
    vecs = embed(list(global_keywords))
    scores_1gram = calculate_label_score(cluster_objs, vecs, global_keywords, lemmatized_qs_list)
    
    
    cluster_objs, lemmatized_qs_list = generate_modified_qs_data(labelling_corpus.documents, clusters, q_ids_list,2)
    global_keywords = generate_global_keywords(labelling_corpus.documents, 2)
    vecs = embed(list(global_keywords))
    scores_2gram = calculate_label_score(cluster_objs, vecs, global_keywords, lemmatized_qs_list)
    
    combined_scores = []
    for score_1gram, score_2gram in zip(scores_1gram, scores_2gram):
        score_1gram.extend(score_2gram)
        score_1gram.sort(key = lambda x: x[1],reverse=True)
        combined_scores.append(score_1gram)

    final_labels = return_best_label_combination(combined_scores)
    return final_labels

"""
    >Used to Check Labels Generated from Label Scores
    >Use this code snippet before returning the labels

    for label, clus_score in zip(final_labels, combined_scores):
        print(clus_score[:5])
        print(label)
        print("-------------")
"""    
    

"""
> The objective of this function is to return whether a label is 1 gram or 2 gram:
1. In a 1 gram, it's one word, with no spaces
2. In a 2 gram, there are two words, with a space in between
4. So this function checks for the space and returns the gram type
5. O/P is 1 for 1 gram & 2 for 2gram
:The inputs required are just the label, which is a string  
"""

def get_label_type(label):
    if ' ' in label:
        return 2 #2gram
    else:
        return 1 #1gram

"""
> The objective of this function is to decide the final labels for clusters:
1. We will only consider 2 Keywords (1 or 2 grams)
2. The algorithm to select these 2 Keywords is basically a set of if-else conditions 
3. It is broken down into 4 cases, based on if the top 2 labels by score are a particular combination of 1 or 2 grams 
4. For each case, certain conditions are checked, to return the best set of labels for that cluster
5. A diagram for this algorithm, created by Araz is found here: 
:The inputs required are the label scores cluster wise 
"""


def return_best_label_combination(label_scores):
    final_labels = []
    for cluster_labels in label_scores:
        label_1 = cluster_labels[0][0]
        label_2 = cluster_labels[1][0]
        label_3 = cluster_labels[2][0]
        label_4 = cluster_labels[3][0]

        label_1_type = get_label_type(label_1)
        label_2_type = get_label_type(label_2)       
        label_3_type = get_label_type(label_3)
        label_4_type = get_label_type(label_4)       

        
        #CASE I label_1 1 gram label_2 2 gram

        if(label_1_type==1 and label_2_type==2):
            if(label_1 not in label_2):
                selected = [label_1, label_2]
                final_labels.append(selected)
            else:
                if(label_3_type==2):
                    selected = [label_2, label_3]
                    final_labels.append(selected)
                else:
                    if(label_3 not in label_2):
                        selected = [label_2, label_3]
                        final_labels.append(selected)
                    else:
                        selected = [label_2, label_4]
                        final_labels.append(selected)

        #CASE II label_1 1 gram label_2 2 gram

        elif(label_1_type==1 and label_2_type==1):
            if(label_3_type==1):
                if(label_4_type==1):
                    selected = [label_1, label_2]
                    final_labels.append(selected)
                else:
                    label_in_word = []
                    if(label_1 in label_4):
                        label_in_word.append(label_1)
                    if(label_2 in label_4):
                        label_in_word.append(label_2)
                    if(label_3 in label_4):
                        label_in_word.append(label_3)
                    
                    if(len(label_in_word)==0):
                        selected = [label_1, label_2]
                        final_labels.append(selected)
                    else:
                        selected = [label_1, label_2, label_3, label_4]
                        for label in label_in_word:
                            if(label in selected):
                                selected.remove(label)
                        final_labels.append(selected)
            else:
                label_in_word = []
                if(label_1 in label_3):
                    label_in_word.append(label_1)
                if(label_2 in label_3):
                    label_in_word.append(label_2)

                if(len(label_in_word)==0):
                    selected = [label_1, label_2]
                    final_labels.append(selected)
                else:
                    if(len(label_in_word)==1):
                        selected = [label_1, label_2, label_3]
                        for label in label_in_word:
                            selected.remove(label)
                        final_labels.append(selected)
                    else:
                        selected = [label_3, label_4]
                        final_labels.append(selected)

        #CASE III label_1 2 gram label_2 1 gram

        elif(label_1_type==2 and label_2_type==1):
            if(label_2 not in label_1):
                selected = [label_1, label_2]
                final_labels.append(selected)
            else:
                if(label_3_type==2):
                    selected = [label_1, label_3]
                    final_labels.append(selected)
                else:
                    if(label_3 not in label_1):
                        selected = [label_1, label_3]
                        final_labels.append(selected)
                    else:
                        selected = [label_1, label_4]
                        final_labels.append(selected)

        #CASE IV label_1 2 gram label_2 2 gram

        else:
            selected = [label_1, label_2]
            final_labels.append(selected)

    return final_labels

"""
> The objective of this function is to generate list of n-grams from a list of words

:The inputs are n: to tell number of grams to take and word_list: list of words to form grams from
"""


def generate_ngrams(n, words_list):
    ngram_set = set()
    for i in range(len(words_list)-n + 1):
        phrase = ' '.join(words_list[i:i+n])
        ngram_set.add(phrase)
    return ngram_set

            
"""
> The objective of this function is to generate:
1. Modified questions formed by joining lemmatized keywords based on number of grams to consider (1 or 2)
    >lemmatized_qs_list
2. Clusters from these modified questions. 
    >modified_clus_list
3. Each lemmatized phrase is added to the global keyword list, which is used for calculation of the NMI score. 
    >global_keywords
4. Calculate the centroids for each cluster, from the USE embeddings
    >centroids
5. Embeddings for all the global keywords
    >vecs
:The inputs required are taken from the inputs of the parent function and an additional 'phrase_len' to tell number of grams  
"""

def generate_modified_qs_data(documents, clusters, q_ids_list, phrase_len):
    #Defining Lists to store required Qs Data
    modified_clus_list = [] #Stores Clusters with modified Qs #ngrams cluster list / cluster list of ngrams
    lemmatized_qs_list = [] #Stores all Questions in Corpus (Modified by joining Keywords in Lemma form)
    centroids = [] # List to store centroid of each cluster
    num_clus = len(clusters) #Total number of Clusters
    cluster_objects = {}

    #Cluster wise data modification and storage
    for cluster_index in range(num_clus):
        cluster_data = {"clus_id":cluster_index}
        clus_q_ids = clusters[cluster_index]
        clus_embeddings = [] #To store existing embeddings of questions for each cluster
        modified_clus_qs = [] #To store new (modified) questions for each cluster
        og_clus_qs = [] #Original Cluster Questions to see for reference with the labels when debugging
        for id in clus_q_ids:
            q_index = q_ids_list.index(id)
            clus_embeddings.append(documents[q_index]._.embedding)
            lemma_q = documents[q_index]._.lemma_list
            n_grams_for_question = generate_ngrams(phrase_len,lemma_q)
            lemmatized_qs_list.append(n_grams_for_question)
            modified_clus_qs.append(n_grams_for_question) #Adding modified question to new formed cluster of modified questions
            og_clus_qs.append(documents[q_index]._.raw_text)
        clus_centroid = np.mean(clus_embeddings, axis=0)
        centroids.append(clus_centroid)
        modified_clus_list.append(modified_clus_qs)
    
        cluster_data["centroid"] = clus_centroid
        cluster_data["keyword_questions"] = modified_clus_qs
        cluster_data["original_clustered_qs"] = og_clus_qs
        cluster_objects[cluster_index] = cluster_data
    
    return cluster_objects, lemmatized_qs_list
    
"""
> The objective of this function is to:
1. Generate list of all unique keywords from corpus
    
2. Keywords are used for labelling, and are of form of n-grams

3. Output is a list of n-gram keywords
    >E.g for 2 grams o/p is a list of all 2-gram keywords of corpus
    
: The inputs required are Documents (Question Objects) and Keyword Length to be considered 
: Keyword Length decides if each keyword is a 1-gram, 2-gram, etc. 
"""

def generate_global_keywords(documents, keyword_length):
    num_docs = len(documents)
    global_keywords = set() #List to store all keywords in corpus (each keyword here is a n-gram, where n is determined by keyword_length)
    for i in range(num_docs):
        logging.debug(documents[i])
        lemma_doc = documents[i]._.lemma_list
        keywords = generate_ngrams(keyword_length, lemma_doc)
        global_keywords.update(keywords)
    global_keywords = set(x.upper() for x in global_keywords)
    return global_keywords



"""
> The objective of this function is to:
1. Calculate the distances for each keyword from the centroid, to be used for scoring
    
2. Call the custom metric function to calculate scores for labels 
    > Call NMI_Metric
    > Add distances term with the factor to normalise, to get the custome score

3. Return list of Top 5 Label Scores for each cluster

:The inputs required are taken from the outputs of the generate_modified_qs_data function 
: The cluster object has attributes: centroid of cluster, raw text questions, lemmatized questions
"""

def calculate_label_score(cluster_objects, vecs, global_keywords, lemmatized_qs_list):
    #For each term and each cluster, get frequencies over all qs present in our corpus (NMI)
    num_clus = len(cluster_objects)
    all_cluster_scores = []
    for cluster_index in range(num_clus):
        modified_clus_qs = cluster_objects[cluster_index]["keyword_questions"] #clus_qs list of qs from the modified cluster 
        og_clus_qs = cluster_objects[cluster_index]["original_clustered_qs"]
        centroid = cluster_objects[cluster_index]["centroid"]
        vec_to_centroid = centroid - vecs
        distances = np.linalg.norm(vec_to_centroid, axis=1)
        logging.debug("Total number of Keywords:",len(distances))
        gloabl_min_distance = min(distances)
        logging.debug("Global Minimum Distance for cluster:",gloabl_min_distance)
    
        NMI_scores = compute_nmi_metrics(global_keywords, lemmatized_qs_list, modified_clus_qs)

        max_NMI = max(NMI_scores,key=lambda x:x[1])[1]
        logging.debug("Maximum NMI Score in Cluster:",max_NMI)
        logging.debug("Length of score list:",len(NMI_scores))
        assert_equal(len(NMI_scores),len(distances)) #Making sure equal lenghts for math to work out xD
        new_score = [[x[0],x[1] + (1/d)*gloabl_min_distance*max_NMI*0.5] for x,d in zip(NMI_scores,distances)]
        new_score.sort(key = lambda x: x[1], reverse=True)
        logging.debug("New Score based Labels:",new_score[:5])
        logging.debug("Cluster Qs:",og_clus_qs)
        all_cluster_scores.append(new_score[:5])
    return all_cluster_scores


"""
> The objective of this function is to:
1. Calculate the Normalised Mutual Information (NMI) score for a term & a cluster
    
2. Append all scores in NMI List for final calculation & comparision
        
:The inputs required are the global keywords list, lemmatized question list, list of cluster questions
and the NMI list, which are modified at each call of this function  
"""


def compute_nmi_metrics(global_keywords, lemmatized_qs_list, clus_qs):
    NMI_scores = []
    total_num_qs = len(lemmatized_qs_list) #Total number of Qs
    for term in global_keywords: #In cluster, calculating I(term,cluster)
        occurences_term_in_cluster = 0 #Number of occurences of the term in the cluster
        occurences_term_not_in_cluster = 0 #Number of occurences of the term not in the cluster
        term_in_cluster = 0 #Bool # To ensure we only calculate %occurences for terms in cluster and to assign 0 scores to terms not in cluster
        P_t0 = 0.0
        P_t1 = 0.0
        P_u0 = 0.0
        P_u1 = 0.0
        P_t0_u0 = 0.0
        P_t0_u1 = 0.0
        P_t1_u0 = 0.0
        P_t1_u1 = 0.0
        for ques in lemmatized_qs_list: #All questions in Corpus
            if term not in ques:
                P_t0+=1
                if(ques not in clus_qs): #clus_qs list of modified qs
                    P_u0+=1
                    P_t0_u0+=1
                else:
                    P_u1+=1
                    P_t0_u1+=1
            else:
                P_t1+=1
                if(ques not in clus_qs):
                    P_u0+=1
                    P_t1_u0+=1
                    occurences_term_not_in_cluster+=1
                else:
                    P_u1+=1
                    P_t1_u1+=1
                    occurences_term_in_cluster+=1
                    term_in_cluster = 1
        if(term_in_cluster): 
            #If term in cluster, from NMI paper making sure that a frequent term in other clusters, but not in this gets negative NMI
            #See page 3 of NMI paper, and reference [1] on footer
            expected_percentage_term_occur_in_cluster = len(clus_qs)/len(lemmatized_qs_list)
            actual_percentage_term_occur_in_cluster = occurences_term_in_cluster/(occurences_term_not_in_cluster + occurences_term_in_cluster)
        I_t_c = 0.0
        P_t0/=total_num_qs
        P_t1/=total_num_qs
        P_u0/=total_num_qs
        P_u1/=total_num_qs
        P_t0_u0/=total_num_qs
        P_t0_u1/=total_num_qs
        P_t1_u0/=total_num_qs
        P_t1_u1/=total_num_qs
        combined_P = {'00': P_t0_u0 ,'01': P_t0_u1 ,'10': P_t1_u0 ,'11': P_t1_u1}
        single_T = {'0': P_t0 ,'1': P_t1}
        single_U = {'0': P_u0 ,'1': P_u1}

        for _ , (key, value) in enumerate(combined_P.items()):
            if(value!=0):
                I_t_c+= (value * math.log((value / (single_T[str(key[0])] * single_U[str(key[1])])),2))
            else:
                I_t_c+=0

        H_t = 0.0
        H_c = 0.0
        for i in range(2):
            if(single_T[str(i)]!=0):
                H_t+=single_T[str(i)]* math.log(single_T[str(i)],2)
            else:
                H_t+=0
            if(single_U[str(i)]!=0):
                H_c+=single_U[str(i)]* math.log(single_U[str(i)],2)
            else:
                H_c+=0
        H_t*=-1
        H_c*=-1
        NMI_i_t_c = 2*(I_t_c/(H_c + H_t))
        if(term_in_cluster):
            if(actual_percentage_term_occur_in_cluster < expected_percentage_term_occur_in_cluster):
                logging.debug("Term where percentage of actual occurences are lesser than expected, if in cluster :", term)
                logging.debug("Actual %:", actual_percentage_term_occur_in_cluster)
                logging.debug("Expected %:", expected_percentage_term_occur_in_cluster)
                NMI_i_t_c*=-1 #As from the paper, to ensure this term doesn't get selected
        else:
            NMI_i_t_c = 0
        NMI_scores.append([term, NMI_i_t_c])
    return NMI_scores
        
        
#Command to use with from terminal
#curl --header "Content-Type:application/json" --request POST --data "@C:\Users\arazs\Documents\Beagle Learning Intern\test_data4.json" http://localhost:5000/usecondition/
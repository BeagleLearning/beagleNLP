import os
import pickle
from nltk.corpus.reader import tagged
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def format_questions(questions, pos, tokenized):

    if len(pos):
        
        tokenized_questions = []
        nlp = spacy.load("en_core_web_sm")
        for ques in questions:
            tokenized = nlp(ques)

            training_q = []
            for token in tokenized:
                if token.pos_ in pos:
                    training_q.append(token.lemma_)
            
            if len(training_q):
                tokenized_questions.append(training_q)

    else:
        tokenized_questions = questions
    
    formatted_questions = []
    if tokenized == False:
        for ques in tokenized_questions:
            q = ' '.join(ques)
            formatted_questions.append(q)
    else:
        formatted_questions = tokenized_questions
    
    return formatted_questions


def get_questions(dataset = 'normal', pos = [], tokenized = True):
    #dataset:
        # normal - 6040 question dataset
        # large - 31000 question dataset
        # mapped - mapped question set
    #pos: List of POS tags which have to be extracted from each question. An empty list indicates no POS filter is necessary.
    #tokenized: Indicates whether the list of questions returned is tokenized or not. 

    #Returns one of three types of values: 
        #A list of questions
        #A list of tokenized questions (list of lists)
        #A dictionary with assignment IDs as keys as lists of questions as values (tokenized or untokenized)

    if len(pos):
        if dataset == 'normal':
            data_path = './Data/Question Sets/6040_question_set'  
        elif dataset == 'large': 
            data_path = './Data/Question Sets/31000_question_set'
        else:
            data_path = './Data/Question Sets/mapped_question_set'

    else:
        if dataset == 'normal':
            data_path = './Data/Question Sets/Preprocessed Question Sets/6040_question_set_preprocessed'
        #this dataset has not been preprocessed yet.    
        elif dataset == 'large': 
            data_path = './Data/Question Sets/31000_question_set'
        else:
            data_path = './Data/Question Sets/Preprocessed Question Sets/mapped_question_set_preprocessed'


        
    with open(data_path,'rb') as infile:
            questions = pickle.load(infile)

    if dataset == 'mapped':
        final_set = {}
        for key in questions:
            final_set[key] = format_questions(questions[key],pos,tokenized)
        return final_set
        
    else:
        return format_questions(questions,pos,tokenized)
            
def display_clusters(questions, clusters, header = ''):

        cluster_numbers = set(clusters)

        #Map questions to the cluster numbers
        mapping = {}
        for i in cluster_numbers:
            mapping[i] = []
    
        for pos in range(0,len(questions)):
            mapping[clusters[pos]].append(questions[pos])
        
        #Print clusters
        print(header)
        print()
        for i in mapping:
            print("Cluster Number:", i)
            for ques in mapping[i]:
                if len(ques):
                    print(ques)
            print()
            print()
            print()

def preprocess(text_list, convert_to_lower = True, lemmatize = True, remove_special_characters = True, remove_stop_words = True, tokenize = False):

    lemmatizer = WordNetLemmatizer()

    custom_stopword_list = ['how','what','when','where','why','who','u']
    stop_words = stopwords.words('english')
    stop_words.extend(custom_stopword_list)

    preprocessed_text_list = []

    for text in text_list:
        if convert_to_lower:
            text = text.lower()
        
        if lemmatize:
            text = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]
        
        if remove_special_characters and remove_stop_words:
            text = [word for word in text if word.isalnum() and word not in stop_words]
        elif remove_special_characters:
            text = [word for word in text if word.isalnum()]
        elif remove_stop_words:
            text = [word for word in text if word not in stop_words]
        
        if not tokenize:
            text = ' '.join(text)
        
        preprocessed_text_list.append(text)
    
    return preprocessed_text_list

def intracluster_similarity(clusters, question_vectors):
    mapping = {}
    for i in range(0,len(clusters)):
        if clusters[i] not in mapping:
            mapping[clusters[i]] = []
        mapping[clusters[i]].append(question_vectors[i])

    print("LEN MAPPING:",len(mapping))
    cosine_similarities = []
    for cluster in mapping:
        cosine_array = cosine_similarity(mapping[cluster])
        vals = []
        for i in range(0,len(mapping[cluster])):
            for j in range(i+1,len(mapping[cluster])):
                vals.append(cosine_array[i,j])

        cosine_similarities.append(np.mean(vals))
    
    return cosine_similarities

def intercluster_similarity(clusters, question_vectors):
    mapping = {}
    for i in range(0,len(clusters)):
        if clusters[i] not in mapping:
            mapping[clusters[i]] = []
        mapping[clusters[i]].append(question_vectors[i])
    
    print("LEN MAPPING:",len(mapping))
    cosine_similarities = {}
    for i in range(0,len(mapping)):
        for j in range(i+1,len(mapping)):
            cosine_array = cosine_similarity(mapping[i],mapping[j])
            row_means = cosine_array.mean(axis = 1)
            cosine_similarities[(i,j)] = np.mean(row_means)
        
    return cosine_similarities


########################################################################################################################################


def display_tags(tagged_questions):
    for question in tagged_questions:
        print(question, ": ",', '.join(tagged_questions[question]))

            
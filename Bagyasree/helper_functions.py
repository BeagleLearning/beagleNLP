import os
import pickle
import spacy


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
                print(ques)
            print()
            print()
            print()
        

            
from testing import Test
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
import pickle
import spacy
import math

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def format_questions_for_BERT(questions, tokenizer):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    
    for ques in questions:
        encoded_dict = tokenizer.encode_plus(
                        ques,                     
                        add_special_tokens = True, 
                        max_length = 32,  
                        pad_to_max_length = True,
                        truncation = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
        
        token_type_ids.append(encoded_dict['token_type_ids'])
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim = 0)
    token_type_ids = torch.cat(token_type_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, token_type_ids

def get_hidden_states_BERT(input_ids, token_type_ids, model):
    model.eval()
    
    hidden_states_master_list = []
    
    #try passing all at once.
    
    with torch.no_grad():
            outputs = model(input_ids,token_type_ids) #the arguments are tensors
            hidden_states_master_list.append(outputs[2])
            
    return hidden_states_master_list

def agg_clustering(no_of_clusters, data):
    cluster = AgglomerativeClustering(n_clusters = no_of_clusters, affinity = 'euclidean', linkage = 'ward')
    clusters = cluster.fit_predict(data)
    return clusters

def display_clusters(clusters, cluster_numbers, questions):

    #Map questions to the cluster numbers
    mapping = {}
    for i in cluster_numbers:
        mapping[i] = []
    
    for pos in range(0,len(questions)):
        mapping[clusters[pos]].append(questions[pos])
    
    #Print clusters
    for i in mapping:
        print("Cluster Number:", i)
        for ques in mapping[i]:
            print(ques)
        print()
        print()
        print()


def transformer_embeddings_agg_clustering(questions):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    inp_ids, tt_ids = format_questions_for_BERT(questions,tokenizer)
    hidden_states = get_hidden_states_BERT(inp_ids, tt_ids, model)
    sentence_embeddings_original = torch.stack(hidden_states[0],dim = 0)
    sentence_embeddings = sentence_embeddings_original.permute(1,0,2,3)
    numpy_embeddings = []
    for sent in sentence_embeddings:
        sent_embedding = torch.mean(sent[-2], dim = 0)
        numpy_embeddings.append(sent_embedding.numpy())
    no_of_clusters = math.floor(math.sqrt(len(questions)))
    res = agg_clustering(no_of_clusters,numpy_embeddings)
    #display_clusters(res, set(res), questions)
    #print(res)
    return res

def get_questions(q_type = '', pos = [], tokenized = True):
    
    if q_type is 'pos':
        with open('questions','rb') as infile:
            spacy_questions = pickle.load(infile)

        nlp = spacy.load("en_core_web_sm")

        questions = []
        for ques in spacy_questions:
            tokenized = nlp(ques)

            training_q = []
            for token in tokenized:
                if token.pos_ in pos:
                    training_q.append(token.lemma_)
            
            if len(training_q):
                questions.append(training_q)
                
    else:
        with open('../preprocessed','rb') as infile:
            questions = pickle.load(infile)
    
    questions_final = []
    
    if tokenized == False:
        for q in questions:
            q = " ".join(q)
            questions_final.append(q)
    else:
        questions_final = questions
    
    return questions_final


if __name__ == '__main__':
    t = Test()
    t.test_clustering_algorithm(transformer_embeddings_agg_clustering)
    # questions = get_questions(tokenized = False)
    # transformer_emebeddings_agg_clustering(questions)
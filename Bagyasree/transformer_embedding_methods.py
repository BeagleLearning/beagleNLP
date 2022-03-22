import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import math
from clustering_methods import ClusteringMethods

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class TransformerEmbeddings: 
    def format_questions_for_BERT(self, questions, tokenizer):
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

    def get_hidden_states_BERT(self, input_ids, token_type_ids, model):
        model.eval()
        
        hidden_states_master_list = []
        
        #try passing all at once.
        
        with torch.no_grad():
                outputs = model(input_ids,token_type_ids) #the arguments are tensors
                hidden_states_master_list.append(outputs[2])
                
        return hidden_states_master_list

    def transformer_embeddings(self, questions):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        inp_ids, tt_ids = self.format_questions_for_BERT(questions,tokenizer)
        hidden_states = self.get_hidden_states_BERT(inp_ids, tt_ids, model)
        sentence_embeddings_original = torch.stack(hidden_states[0],dim = 0)
        sentence_embeddings = sentence_embeddings_original.permute(1,0,2,3)
        numpy_embeddings = []
        for sent in sentence_embeddings:
            sent_embedding = torch.mean(sent[-2], dim = 0)
            numpy_embeddings.append(sent_embedding.numpy())
        no_of_clusters = math.floor(math.sqrt(len(questions)))
        clusters = ClusteringMethods.agglomerative_clustering(numpy_embeddings, no_of_clusters)
        return clusters
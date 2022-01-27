import os
import tensorflow_hub as hub
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordfreq import word_frequency
import lda
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.naive_bayes import  ComplementNB

CUSTOM_STOPWORD_LIST = ['how','what','when','where','why','who','u','does','do', 'would','many']

#Defines the distance threshold in agglomerative_clustering.
DISTANCE_THRESHOLD = 1.35

#Initial number of tags to extract from question set before further pruning, in get_top_n_keywords.
CANDIDATE_TAG_NUM = 40
#Fraction of range, used for computing the threshold in complement_naive_bayes.
FRACTION = 0.3
#Minimum number of questions to assign to each tag in complement_naive_bayes.
N = 2

class General:
    def preprocess(self, text_list, convert_to_lower = True, lemmatize = True, remove_special_characters = True, remove_stop_words = True, tokenize = False):
        '''
            Performs basic preprocessing and cleaning operations on a list of strings. 

            Input:
                text_list: List of strings.
                The other parameters let you decide what sort of preprocessing you want for text_list.
            
            Output:
                preprocessed_text_list: List of preprocessed strings.
        '''
        try:
            lemmatizer = WordNetLemmatizer()

            stop_words = stopwords.words('english')
            stop_words.extend(CUSTOM_STOPWORD_LIST)

            preprocessed_text_list = []

            for text in text_list:
                if convert_to_lower:
                    text = text.lower()

                text = [word for word in word_tokenize(text)]

                if remove_stop_words:
                    text = [word for word in text if word not in stop_words]
                if remove_special_characters:
                    text = [word for word in text if word.isalnum()]
                if lemmatize:
                    text = [lemmatizer.lemmatize(word) for word in text]
                if not tokenize:
                    text = ' '.join(text)
                
                preprocessed_text_list.append(text)
            
            return preprocessed_text_list
        
        except Exception as e:
            print('Error: ',e)
            raise e

    def universal_sentence_encoder(self, dataset, return_embed = False):
        '''
            Embeds the dataset based on USE. 

            Input:
                dataset: List of strings
                return_embed: Set to True if you want the embedder to be returned for further use.
            
            Output:
                vectors: List of embeddings
                embed: Embedder; returned if return_embed is True.
            
            Note: Change the environment file path to a local path. 
        '''

        try:

            dir_path = os.path.dirname(os.path.realpath(__file__))
            os.environ['TFHUB_CACHE_DIR'] = dir_path
            embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 
            vectors = embed(dataset)

            if return_embed:
                return vectors, embed
            else:
                return vectors
        
        except Exception as e:
            print('Error: ',e)
            raise e

    def lda_merge(self, labelled_questions, return_group_tags, question_number_ratio):
        '''
            Merges groups of questions based on similarity of questions within the groups using LDA.

            Input: 
                labelled_questions: A dictionary of labels with values as lists of questions. Each key-value pair is treated as one group.
                return_group_tags: If true, the returned dictionary has tuples of the original labels as keys. 
            
            Output: 
                return_dict: A dictionary in which each value is a list of questions. 
                If return_group_tags is true, the keys will be groupings of the original labels. If not, the labels will be reset.
        '''
        try:

            labels = list(labelled_questions.keys())
            #Create a list of strings where each string is concatenation of all questions in a group. 
            label_docs = [' '.join(labelled_questions[label]) for label in labels]

            #Fit an LDA model on the concatenated strings based on token counts and generate topics.
            vec = CountVectorizer(analyzer='word', ngram_range=(1,3))
            X = vec.fit_transform(label_docs)
            model = lda.LDA(n_topics=math.ceil(question_number_ratio*len(labels)), random_state=1)
            model.fit(X)
            doc_topic = model.doc_topic_

            #Find the topic to which each label belongs and create a dictionary of label clusters.
            label_clusters = {}
            for i in range(len(label_docs)):
                topic = doc_topic[i].argmax()
                if topic not in label_clusters:
                    label_clusters[topic] = []
                label_clusters[topic].append(labels[i])
            
            merged_question_groups = {}
            
            #Put all questions belonging to a cluster of labels in the same group.
            for topic in label_clusters:
                labels = label_clusters[topic]
                question_set = set([question for label in labels for question in labelled_questions[label]])
                merged_question_groups[topic] = list(question_set)
            
            
            if(return_group_tags):
                #Assign topic names as tuples of labels in the cluster. 
                tagged_groups = {}
                for topic in merged_question_groups:
                    tags = label_clusters[topic]
                    tagged_groups[tuple(tags)] = merged_question_groups[topic]
                return_dict = tagged_groups
            else:
                return_dict = merged_question_groups
            
            return return_dict

        except Exception as e:
            print('Error: ',e)
            raise e
            

class Clustering:
    def agglomerative_clustering(self, vectors):
        '''
            Performs agglomerative clustering on a set of embeddings. 

            Input: 
                vectors: List of embeddings, where each embedding corresponds to a question.
            
            Output: 
                clusters: List of cluster labels, where each label corresponds to an embedding.
        '''
        try:
            #Distance threshold: Distance up to which merging happens.
            cluster = AgglomerativeClustering(n_clusters = None, affinity = 'euclidean', linkage = 'ward', distance_threshold=DISTANCE_THRESHOLD, compute_full_tree=True)
            clusters = cluster.fit_predict(vectors)
            return clusters
        
        except Exception as e:
            raise e
    

class Tagging: 
    def get_top_n_keywords(self, n, questions):
        '''
            Finds the keywords in the set of questions that can be used as tags. 

            Input:
                n: Number of keywords to fetch. 
                questions: List of strings.
            
            Output: 
                top_n: List of strings where each string can be treated as a tag.
        '''
        try:
            #Get the TF-IDF matrix for the questions.
            preprocessed_questions = General().preprocess(questions, tokenize=False)
            vectorizer = TfidfVectorizer(ngram_range=(1,2))
            doc_term_matrix = vectorizer.fit_transform(preprocessed_questions)
            all_terms = vectorizer.get_feature_names_out()
            doc_term_array = doc_term_matrix.toarray()
            
            term_scores = np.array([0] * len(all_terms))

            # Fetch the CANDIDATE_TAG_NUM highest scoring terms by adding the scores received by the term for all questions in the tf-idf matrix.
            for i in range(0, len(questions)):
                doc = doc_term_array[i]
                for term_index in range(0, len(doc)):
                    term_scores[term_index] += doc[term_index]
            
            top_terms = (-term_scores).argsort()[:CANDIDATE_TAG_NUM]

            #Find the frequency of the top CANDIDATE_TAG_NUM terms in the English language
            freqs = np.array()
            for term_index in top_terms:
                try:
                    freqs.append(word_frequency(all_terms[term_index],"en"))
                except:
                    pass

            #Fetch the top n rarest terms in the English language out of the set of 40 terms.
            top_terms_by_frequency = (freqs).argsort()[:n]
            roots = []
            for term in top_terms_by_frequency:
                roots.append(all_terms[top_terms[term]])
            
            
            return top_terms_by_frequency
        
        except Exception as e:
            print('Error: ',e)
            raise e

    def complement_naive_bayes(self, roots, questions):
        '''
            Maps tags to question embeddings based on the Complement Naive Bayes algorithm. 

            Input: 
                questions: List of strings.
                roots: List of strings, where each string is a tag.
            
            Output: 
                root_to_ques_dict: A dictionary with tags as keys and a lists of questions as values. It indicates which questions can be mapped to which tags.
        '''
        try:
            embeddings, embed = General().universal_sentence_encoder(questions, True)
            #Add 1 to each value because CNB cannot work with negative values, and the embedding values never go below -1.
            adjusted_embeddings = embeddings + 1
            question_ids = [i for i in range(0, len(questions))]
            cnb_clf = ComplementNB(alpha=1, fit_prior=True)
            cnb_clf.fit(adjusted_embeddings, question_ids)
        
            root_to_ques_dict = {}
            for root in roots:
                    root_embedding = embed([root]) + 1

                    #Get the probability for each question belonging to the root.
                    probabilities = cnb_clf.predict_proba(root_embedding)[0]

                    #Find the top FRACTION of questions based on highest probabilites. 
                    max = np.max(probabilities)
                    min = np.min(probabilities)
                    threshold = max - (max - min) * FRACTION
                    questions_above_threshold = [questions[i] for i in range(0, len(probabilities)) if probabilities[i] > threshold]

                    #Find top N questions
                    question_indices_by_probability = np.argsort(-probabilities)[:N]
                    top_n_questions = [questions[index] for index in question_indices_by_probability]

                    #Combine set of questions above the threshold and top N questions. This guarantees a minimum number of questions for the tag. 
                    #If we want only the questions whose probabilities are above the threshold, set root_questions = questions_above_threshold.
                    root_questions = list(set(questions_above_threshold + top_n_questions))
                    root_to_ques_dict[root] = root_questions

            return root_to_ques_dict
        
        except Exception as e:
            print('Error: ',e)
            raise e
        
    def map_tags(self, tag_dict):
        '''
            Maps a tuple of terms into a representative single term.

            Input: 
                tag_dict: A dictionary whose keys are tuples of strings.
            
            Output: 
                mapped_dict: A dictionary whose keys are strings.
        '''
        try:
            mapped_dict = {}
            for tag_tuple in tag_dict:
                tag_string = tag_tuple[0] #Replace with a mapping algorithm
                mapped_dict[tag_string] = tag_dict[tag_tuple]

            return mapped_dict
        
        except Exception as e:
            print('Error: ',e)
            raise e


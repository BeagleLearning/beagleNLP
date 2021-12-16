import sys
sys.path.append('../')
from nltk.corpus.reader import tagged
from Bagyasree.Testing.testing import Test
from Question_Categorization.clustering_and_tagging_classes import Clustering, Tagging, General

class FinalAlgorithms:

    def get_question_strings(self, questions):
        try:
            #Create arrays of question IDs and question strings with matching indices.
            question_ids = []
            question_strings = []
            for question in questions:
                question_ids.append(question['id'])
                question_strings.append(question['text'])
            
            return question_ids, question_strings
        
        except Exception as e:
            print('Error: ',e)
            raise e

    #Agglomerative clustering using Universal Sentence Encoding
    def clustering(self, questions):

        try:

            question_ids, question_strings = self.get_question_strings(questions)

            #Algorithm on question strings.
            vectors = General().universal_sentence_encoder(question_strings)
            cluster_mapping = Clustering().agglomerative_clustering(vectors)

            #Match question IDs to clusters.
            clusters= {}
            for i in range(0, len(cluster_mapping)):
                key_string = str(cluster_mapping[i])
                if key_string not in clusters:
                    clusters[key_string] = []
                clusters[key_string].append(str(question_ids[i]))
            
            #Format output.
            output = []
            for cluster in clusters:
                output.append({'label':cluster, 'questions':clusters[cluster]})
            
            return output
        
        except Exception as e:
            print('Error: ',e)
            raise e
            
    #Custom algorithm using Complement Naive Bayes and LDA
    def tagging(self, questions):
        try:
            question_ids, question_strings = self.get_question_strings(questions)

            tagging_class = Tagging()

            #Find the top n terms in the questions that can be tags. The final dictionary will contain 0.7*n tags or less. (n<=40)
            keywords = tagging_class.get_top_n_keywords(15, question_strings)
            print('Got tags.')
            #Map questions to tags.
            tag_dict = tagging_class.complement_naive_bayes(keywords, question_strings)
            print('Mapped questions to tags..')
            #Merge tags with similar sets of questions. Resulting dict has tuples as keys.
            merged_tag_dict = General().lda_merge(tag_dict, return_group_tags=True)
            print('Merged groups.')
            #Map a tuple of tags to one single string.
            merged_tag_dict = tagging_class.map_tags(merged_tag_dict)
            print('Mapped tuples.')
            #Match question IDs to tags.
            tagged_questions = {}
            for tag in merged_tag_dict:
                id_list = [str(question_ids[question_strings.index(question_string)]) for question_string in merged_tag_dict[tag]]
                tagged_questions[tag] = id_list
            
            #Format output.
            output = []
            for tag in tagged_questions:
                output.append({'label':tag, 'questions':tagged_questions[tag]})
            
            print(output)
            return output
        
        except Exception as e:
            print('Error: ',e)
            raise e




import sys
sys.path.append('../')
from Question_Categorization.clustering_and_tagging_classes import Clustering, Tagging, General

class FinalAlgorithms:

    def get_question_strings(self, questions, ids = False):
        try:
            #Create arrays of question IDs and question strings with matching indices.
            question_strings =  list(map(lambda question: question["text"], questions))
            if ids:
                question_ids = list(map(lambda question: question["id"], questions))
                return question_strings, question_ids

            return question_strings
        
        except Exception as e:
            err = 'Error in FinalAlgorithms.get_question_strings: ' + str(e)
            raise err

    #Agglomerative clustering using Universal Sentence Encoding
    def clustering(self, questions):

        try:

            # question_ids, question_strings = self.get_question_strings(questions)
            question_strings = self.get_question_strings(questions)

            #Algorithm on question strings.
            vectors = General().universal_sentence_encoder(question_strings)
            cluster_mapping = Clustering().agglomerative_clustering(vectors)

            #Match question IDs to clusters.
            clusters= {}
            # for i in range(0, len(cluster_mapping)):
            for i, cluster_label in enumerate(cluster_mapping):
                # key_string = str(cluster_mapping[i])
                if cluster_label not in clusters:
                    clusters[cluster_label] = []
                clusters[cluster_label].append(str(questions[i]['id']))
            
            #Format output.
            output = []
            for cluster_label in clusters:
                output.append({'label':str(cluster_label), 'questions':clusters[cluster_label]})
            
            return output
        
        except Exception as e:
            err = 'Error in FinalAlgorithms.clustering: ' + str(e)
            raise Exception(err)
            
    #Custom algorithm using Complement Naive Bayes and LDA
    def tagging(self, questions):
        try:
            question_strings, question_ids = self.get_question_strings(questions, ids = True)

            tagging_class = Tagging()

            #Find the top n terms in the questions that can be tags. The final dictionary will contain 0.7*n tags or less. (n<=40)
            keywords = tagging_class.get_top_n_keywords(0.25, question_strings)
            
            #Map questions to tags.
            tag_dict = tagging_class.complement_naive_bayes(keywords, question_strings)
            
            #Merge tags with similar sets of questions. Resulting dict has tuples as keys.
            merged_tag_dict = General().lda_merge(tag_dict, return_group_tags=True, question_number_ratio=0.7)
            
            #Map a tuple of tags to one single string.
            embeddings, embed = General().universal_sentence_encoder(question_strings, return_embed=True)
            merged_tag_dict = tagging_class.map_tags(merged_tag_dict, embed)
            
            #Match question IDs to tags.
            tagged_questions = {}
            for tag in merged_tag_dict:
                id_list = [str(question_ids[question_strings.index(question_string)]) for question_string in merged_tag_dict[tag]]
                tagged_questions[tag] = id_list
            
            #Format output.
            output = []
            for tag in tagged_questions:
                output.append({'label':tag, 'questions':tagged_questions[tag]})
            
            return output
        
        except Exception as e:
            err = 'Error in FinalAlgorithms.tagging: ' + str(e)
            raise Exception(err)




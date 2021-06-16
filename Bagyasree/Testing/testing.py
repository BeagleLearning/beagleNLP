#Testing measures: Extrinsic and Intrinsic. 
#Extrinsic: Adjusted Rand Index, Fowlkes-Mallows Score, Normalised Mutual Information Score
#Intrinsic: High intracluster similarity and low intercluster similarity (Not implemented yet).

import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score

test_data_path = './Testing/test_data.csv'

class Test:

    def get_cluster_set_mapping(self,df):
        df = df.fillna('NULL')
        list_of_tags = list(df)
        cluster_mapping = []
        cluster_keys = list(df.unique())

        null_count = list_of_tags.count('NULL')

        if null_count < len(list_of_tags)/2:
            for tag in list_of_tags: 
                #print(tag, cluster_keys.index(tag))
                cluster_mapping.append(cluster_keys.index(tag))
        
        return cluster_mapping


    def __init__(self):
        test_data = pd.read_csv(test_data_path)
        
        grouped_data = test_data.groupby('File_ID')

        #Dictionary with file IDs as keys; each file ID has a list of questions associated with it.
        self.test_questions = {}

        #Dictionary with file IDs as keys; each file ID has a list associated with it.
        #Each list contains one list of cluster mappings for each set of tags in the test dataset.
        self.cluster_sets = {}

        #Dictionary with file IDs as keys; each file ID has a dictionary associated with it.
        #Each dictionary contains a (question, list of tags) pair for each question ID in the file.
        self.tag_sets = {}

        col_names = []
        for column in test_data.columns:
            if 'Tag' in column:
                col_names.append(column)
            
        for key, group_info in grouped_data:
            group = grouped_data.get_group(key).reset_index()
            df = group[col_names]
            self.test_questions[key] = list(group['Question'])
            self.cluster_sets[key] = []
            self.tag_sets[key] = {}

            for col in col_names:
                cluster_df = df[col]
                cluster_set = self.get_cluster_set_mapping(cluster_df)
                self.cluster_sets[key].append(cluster_set)
        
            for index, row in group.iterrows():
                tag_list = [row[col] for col in col_names]
                self.tag_sets[key][row['ID']] = (row['Question'],tag_list)


    
    def get_test_questions(self, print_questions = False):
        if print_questions: 
            for file_id in self.test_questions:
                print("FILE ID:",file_id)
                for ques in self.test_questions[file_id]:
                    print(ques)
                print()
                print()
        return self.test_questions
    
    def get_tag_sets(self, print_sets = False):
        if print_sets:
            for file_id in self.tag_sets:
                print("FILE ID:", file_id)
                for ques in self.tag_sets[file_id]:
                    print(self[file_id][ques])
                print()
                print()
        return self.tag_sets
    
    def get_cluster_sets(self, print_sets = False):
        if print_sets:
            for file_id in self.cluster_sets:
                print("FILE ID:", file_id)
                for cluster_mapping in self.cluster_sets[file_id]:
                    print(cluster_mapping)
                print()
                print()
        return self.cluster_sets


    def display_clusters(self, questions, clusters, header):

        cluster_numbers = set(clusters)

        #Map questions to the cluster numbers
        mapping = {}
        for i in cluster_numbers:
            mapping[i] = []
    
        for pos in range(0,len(questions)):
            mapping[clusters[pos]].append(questions[pos])
        
        #Print clusters
        print("SET ",header)
        print()
        for i in mapping:
            print("Cluster Number:", i)
            for ques in mapping[i]:
                print(ques)
            print()
            print()
            print()


    def compute_clustering_similarity(self, predicted_clusters, true_cluster_sets):
        '''
            predicted_clusters: Array of cluster numbers corresponding to each question, predicted by the algorithm.
            true_clusters: Array of actual cluster numbers for each question.
        '''
        #Arrays of scores. Each value corresponds to the value obtained for one cluster set (i.e., one column in the test dataset).
        scores_rand = [] 
        scores_nmi = []
        scores_fm = []
        for true_clusters in true_cluster_sets:
            if(len(true_clusters)):
                score_rand = adjusted_rand_score(predicted_clusters, true_clusters)
                scores_rand.append(score_rand)
                score_nmi = normalized_mutual_info_score(true_clusters, predicted_clusters)
                scores_nmi.append(score_nmi)
                score_fm = fowlkes_mallows_score(true_clusters, predicted_clusters)
                scores_fm.append(score_fm)
            else:
                scores_rand.append('NA')
                scores_nmi.append('NA')
                scores_fm.append('NA')
        return scores_rand, scores_nmi, scores_fm


    def test_clustering_algorithm(self, algorithm, arguments = [], display_clusters = False):
        '''
            algorithm: A clustering function which takes in a list of questions as a parameter, and returns a list with cluster numbers corresponding to each question.
            arguments: A list of optional arguments to be passed to the algorithm. 
        '''
        question_sets = self.get_test_questions()
        cluster_sets = self.get_cluster_sets()

        all_scores_rand = {}
        all_scores_nmi = {}
        all_scores_fm = {}
        no_of_sets = 0
        
        for key in question_sets:
            list_of_questions = question_sets[key]
            result = algorithm(list_of_questions, *arguments)
            scores_rand, scores_nmi, scores_fm = self.compute_clustering_similarity(result,cluster_sets[key])
            all_scores_rand[key] = scores_rand
            all_scores_nmi[key] = scores_nmi
            all_scores_fm[key] = scores_fm
            no_of_sets = len(scores_rand)
        
            if display_clusters:
                self.display_clusters(list_of_questions, result, key)

        rand_scores_df = pd.DataFrame.from_dict(all_scores_rand, orient="index")
        rand_scores_df.columns = ['Cluster Set '+str(i) for i in range(1,no_of_sets+1)]
        print("RAND INDEX SCORES:")
        print(rand_scores_df)
        print()

        fm_scores_df = pd.DataFrame.from_dict(all_scores_fm, orient="index")
        fm_scores_df.columns = ['Cluster Set '+str(i) for i in range(1,no_of_sets+1)]
        print("FOWLKES-MALLOWS SCORES:")
        print(fm_scores_df)
        print()

        nmi_scores_df = pd.DataFrame.from_dict(all_scores_nmi, orient="index")
        nmi_scores_df.columns = ['Cluster Set '+str(i) for i in range(1,no_of_sets+1)]
        print("NORMALIZED MUTUAL INFORMATION SCORES:")
        print(nmi_scores_df)

        
            
    
    #Does not work at present.
    def test_tagging_algorithm(self, algorithm):
        '''
            algorithm: A tagging function which takes in a list of questions as a parameter, and returns a list of tags for each question.
        '''
        question_sets = self.get_test_questions()
        for key in question_sets:
            list_of_questions = question_sets[key]
            result = algorithm(list_of_questions)

if __name__ == '__main__':
    t = Test()
    #cluster_sets = t.get_cluster_sets(print_sets=True)
    
import unittest
from unittest.mock import patch 

import os,sys,inspect

from numpy.testing._private.utils import assert_equal
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from use_cluster import get_data_embeddings, get_ind, get_clusters_from_sim_matrix, HAC_with_Sparsification, best_score_HAC_sparse, get_best_HAC_normal,return_cluster_dict
from beagleError import BeagleError
import errors
import numpy as np


class TestUSEClustering(unittest.TestCase):
    def setUp(self): #Runs before every test
        self.json_data = [{"question":"How is social distancing going to impact the economy","question_id":2,"course_id":"test"},
        {"question":"Does helping people out of poverty help the economy","question_id":2,"course_id":"test"},
        {"question":"How is supply and demand being affected with the current pandemic","question_id":2,"course_id":"test"},
        {"question":"is capitalism the most successful economic system","question_id":2,"course_id":"test"},
        {"question":"How does the unemployment rate effect the economy","question_id":2,"course_id":"test"},
        {"question":"If the federal government raised minimum wage, would it positively effect the national economy","question_id":2,"course_id":"test"},
        {"question":"How would a free market system affect the private sector","question_id":2,"course_id":"test"},
        {"question":"How will you reach consensus on the definition of what is an equitable economic system","question_id":2,"course_id":"test"},
        {"question":"What country has been most successful with their economy and how did they go about it","question_id":2,"course_id":"test"},
        {"question":"What has the unemployment rate, due to the pandemic, done for our economic system and society How has the closers on the food industries, due to the pandemic, affected our economic system and society","question_id":2,"course_id":"test"}]

    """ Testing the Get Data Embeddings Function
    1. Normal Case Test
    2. 3 Key Errors
    3. 3 Invalid Datatype Errors
    """

    def test_valid_get_data_embeddings(self):
        #Test with Valid Data
        self.assertEqual(get_data_embeddings(self.json_data)[1:],(['How is social distancing going to impact the economy', 'Does helping people out of poverty help the economy', 'How is supply and demand being affected with the current pandemic', 'is capitalism the most successful economic system', 'How does the unemployment rate effect the economy', 'If the federal government raised minimum wage, would it positively effect the national economy', 'How would a free market system affect the private sector', 'How will you reach consensus on the definition of what is an equitable economic system', 'What country has been most successful with their economy and how did they go about it', 'What has the unemployment rate, due to the pandemic, done for our economic system and society How has the closers on the food industries, due to the pandemic, affected our economic system and society'],[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
        self.assertEqual(len(get_data_embeddings(self.json_data)[0][0]), 512)

    #Testing for Various Key errors
    def test_ques_keyerror_getdataembeddings(self):
        
        #Missing key: 'Question'
        del self.json_data[0]['question']
        
        with self.assertRaises(BeagleError) as an_error:
            get_data_embeddings(self.json_data)

        er_raised = an_error.exception
        self.assertEqual(er_raised.code,2805) #confirming it's a Key Error

    def test_quesid_keyerror_getdataembeddings(self):
        
        #Missing key: 'Question Id'
        del self.json_data[0]['question_id']
        
        with self.assertRaises(BeagleError) as an_error:
            get_data_embeddings(self.json_data)

        er_raised = an_error.exception
        self.assertEqual(er_raised.code,2805) #confirming it's a Key Error

    def test_courseid_keyerror_getdataembeddings(self):
        
        #Missing key: 'Course ID'
        del self.json_data[0]['course_id']
        
        with self.assertRaises(BeagleError) as an_error:
            get_data_embeddings(self.json_data)

        er_raised = an_error.exception
        self.assertEqual(er_raised.code,2805) #confirming it's a Key Error

    #Testing for Invalid Datatype Errors
    def test_question_datatype_error_getdataembeddings(self):
        
        #Invalid Datatype for 'Question'
        #Datatype Accepted: str

        self.json_data[0]['question'] = 4
        
        with self.assertRaises(BeagleError) as an_error:
            get_data_embeddings(self.json_data)

        er_raised = an_error.exception
        self.assertEqual(er_raised.code,2807) #confirming it's an Invalid Datatype Error

        self.json_data[0]['question'] = ["Test"]
        
        with self.assertRaises(BeagleError) as an_error:
            get_data_embeddings(self.json_data)

        er_raised = an_error.exception
        self.assertEqual(er_raised.code,2807) #confirming it's an Invalid Datatype Error


    def test_questionid_datatype_error_getdataembeddings(self):
        
        #Invalid Datatype for 'Question'
        #Datatype Accepted: str

        self.json_data[0]['question_id'] = "4"
        
        with self.assertRaises(BeagleError) as an_error:
            get_data_embeddings(self.json_data)

        er_raised = an_error.exception
        self.assertEqual(er_raised.code,2807) #confirming it's an Invalid Datatype Error

        self.json_data[0]['question_id'] = [4]
        
        with self.assertRaises(BeagleError) as an_error:
            get_data_embeddings(self.json_data)

        er_raised = an_error.exception
        self.assertEqual(er_raised.code,2807) #confirming it's an Invalid Datatype Error


    def test_courseid_datatype_error_getdataembeddings(self):
        
        #Invalid Datatype for 'Question'
        #Datatype Accepted: str

        self.json_data[0]['course_id'] = {5:"5"}
        
        with self.assertRaises(BeagleError) as an_error:
            get_data_embeddings(self.json_data)

        er_raised = an_error.exception
        self.assertEqual(er_raised.code,2807) #confirming it's an Invalid Datatype Error

        self.json_data[0]['course_id'] = [4]
        
        with self.assertRaises(BeagleError) as an_error:
            get_data_embeddings(self.json_data)

        er_raised = an_error.exception
        self.assertEqual(er_raised.code,2807) #confirming it's an Invalid Datatype Error

    """Testing the Get Ind Function: All Normal & Edge Cases"""

    def test_getind(self):
        
        #Testing some normal cases
        
        self.assertEqual(get_ind(3,3),(1,2))
        self.assertEqual(get_ind(4,6),(2,3))
        self.assertEqual(get_ind(6,3),(0,3))
        self.assertEqual(get_ind(5,10),(3,4))
        self.assertEqual(get_ind(7,7),(1,2))


    """Testing the get clusters from sim matrix Function"""

    def test_getclusfrom_sim_matrix(self):
        
        #Testing some normal cases
        _,data_used,_ = get_data_embeddings(self.json_data)
        tst_X = [[ 0.0000000e+00 ,5.8276880e-01,  6.6471207e-01,  1.0000000e+00,
        4.0161836e-01 , 6.0324907e-01 , 5.2772105e-01 , 6.8446219e-01,
        1.0000000e+00 , 5.8776689e-01],
        [ 5.8276880e-01 ,-1.1920929e-07,  1.0000000e+00,  6.0412520e-01,
        5.3194338e-01 , 5.3522438e-01 , 6.6648048e-01 , 1.0000000e+00,
        1.0000000e+00,  1.0000000e+00],
        [ 6.6471207e-01 , 1.0000000e+00 , 1.1920929e-07 , 1.0000000e+00,
        5.8649647e-01 , 1.0000000e+00 , 6.8129843e-01 , 1.0000000e+00,
        1.0000000e+00 , 4.0862775e-01],
        [ 1.0000000e+00 , 6.0412520e-01 , 1.0000000e+00 , 1.1920929e-07,
        1.0000000e+00 , 1.0000000e+00 , 6.4433789e-01 , 6.5130597e-01,
        4.6070302e-01 , 1.0000000e+00],
        [ 4.0161836e-01 , 5.3194338e-01 , 5.8649647e-01  ,1.0000000e+00,
        2.3841858e-07 , 5.0259250e-01 , 5.6211746e-01 , 1.0000000e+00,
        7.3671132e-01 , 3.2635254e-01],
        [ 6.0324907e-01 , 5.3522438e-01 , 1.0000000e+00 , 1.0000000e+00,
        5.0259250e-01, -1.1920929e-07,  5.1762718e-01,  1.0000000e+00,
        1.0000000e+00,  1.0000000e+00],
        [ 5.2772105e-01 , 6.6648048e-01 , 6.8129843e-01 , 6.4433789e-01,
        5.6211746e-01,  5.1762718e-01 ,-2.3841858e-07,  6.4054787e-01,
        1.0000000e+00 , 1.0000000e+00],
        [ 6.8446219e-01 , 1.0000000e+00 , 1.0000000e+00 , 6.5130597e-01,
        1.0000000e+00 , 1.0000000e+00 , 6.4054787e-01 , 0.0000000e+00,
        1.0000000e+00 , 1.0000000e+00],
        [ 1.0000000e+00 , 1.0000000e+00 , 1.0000000e+00 , 4.6070302e-01,
        7.3671132e-01 , 1.0000000e+00 , 1.0000000e+00 , 1.0000000e+00,
        -1.1920929e-07 , 7.2638398e-01],
        [ 5.8776689e-01 , 1.0000000e+00 , 4.0862775e-01 , 1.0000000e+00,
        3.2635254e-01 , 1.0000000e+00 , 1.0000000e+00 , 1.0000000e+00,
        7.2638398e-01 ,-2.3841858e-07]]
        self.assertEqual(get_clusters_from_sim_matrix(4,data_used,tst_X)[0],
        {0: ['How is social distancing going to impact the economy', 'Does helping people out of poverty help the economy', 'How does the unemployment rate effect the economy', 'If the federal government raised minimum wage, would it positively effect the national economy', 'How would a free market system affect the private sector'], 1: ['How is supply and demand being affected with the current pandemic', 'What has the unemployment rate, due to the pandemic, done for our economic system and society How has the closers on the food industries, due to the pandemic, affected our economic system and society'], 2: ['is capitalism the most successful economic system', 'What country has been most successful with their economy and how did they go about it'], 3: ['How will you reach consensus on the definition of what is an equitable economic system']})
        self.assertEqual([int(x) for x in get_clusters_from_sim_matrix(4,data_used,tst_X)[1]],[0,0,1,2,0,0,0,3,2,1])
        
    """ Testing the HAC with Sparsification Function:
    1. If correct Similarity Matrix is produced
    2. If Q_Clusters are generated correctly
    3. If Correct Groups are generated"""

    def test_HACwithSpars(self):
        
        #Testing Correct Similarity Matrix is generated
        #Testing correct q_clusters
        #Testing correct groups

        embed, data_used, _ = get_data_embeddings(self.json_data)

        tst_X = np.array([[ 0.0000000e+00 ,5.8276880e-01,  6.6471207e-01,  1.0000000e+00,
        4.0161836e-01 , 6.0324907e-01 , 5.2772105e-01 , 6.8446219e-01,
        1.0000000e+00 , 5.8776689e-01],
        [ 5.8276880e-01 ,-1.1920929e-07,  1.0000000e+00,  6.0412520e-01,
        5.3194338e-01 , 5.3522438e-01 , 6.6648048e-01 , 1.0000000e+00,
        1.0000000e+00,  1.0000000e+00],
        [ 6.6471207e-01 , 1.0000000e+00 , 1.1920929e-07 , 1.0000000e+00,
        5.8649647e-01 , 1.0000000e+00 , 6.8129843e-01 , 1.0000000e+00,
        1.0000000e+00 , 4.0862775e-01],
        [ 1.0000000e+00 , 6.0412520e-01 , 1.0000000e+00 , 1.1920929e-07,
        1.0000000e+00 , 1.0000000e+00 , 6.4433789e-01 , 6.5130597e-01,
        4.6070302e-01 , 1.0000000e+00],
        [ 4.0161836e-01 , 5.3194338e-01 , 5.8649647e-01  ,1.0000000e+00,
        2.3841858e-07 , 5.0259250e-01 , 5.6211746e-01 , 1.0000000e+00,
        7.3671132e-01 , 3.2635254e-01],
        [ 6.0324907e-01 , 5.3522438e-01 , 1.0000000e+00 , 1.0000000e+00,
        5.0259250e-01, -1.1920929e-07,  5.1762718e-01,  1.0000000e+00,
        1.0000000e+00,  1.0000000e+00],
        [ 5.2772105e-01 , 6.6648048e-01 , 6.8129843e-01 , 6.4433789e-01,
        5.6211746e-01,  5.1762718e-01 ,-2.3841858e-07,  6.4054787e-01,
        1.0000000e+00 , 1.0000000e+00],
        [ 6.8446219e-01 , 1.0000000e+00 , 1.0000000e+00 , 6.5130597e-01,
        1.0000000e+00 , 1.0000000e+00 , 6.4054787e-01 , 0.0000000e+00,
        1.0000000e+00 , 1.0000000e+00],
        [ 1.0000000e+00 , 1.0000000e+00 , 1.0000000e+00 , 4.6070302e-01,
        7.3671132e-01 , 1.0000000e+00 , 1.0000000e+00 , 1.0000000e+00,
        -1.1920929e-07 , 7.2638398e-01],
        [ 5.8776689e-01 , 1.0000000e+00 , 4.0862775e-01 , 1.0000000e+00,
        3.2635254e-01 , 1.0000000e+00 , 1.0000000e+00 , 1.0000000e+00,
        7.2638398e-01 ,-2.3841858e-07]])
        
        #print(type(tst_X))
        #print(type(tst_X[0]))
        #Testing if correct matrix is returned
        grouped, q_clus, X = HAC_with_Sparsification(4, embed, data_used, 2, 1)
        #print(X)
        self.assertEqual(X.all(),tst_X.all())
        self.assertEqual(grouped,{0: ['How is social distancing going to impact the economy', 'Does helping people out of poverty help the economy', 'How does the unemployment rate effect the economy', 'If the federal government raised minimum wage, would it positively effect the national economy', 'How would a free market system affect the private sector'], 1: ['How is supply and demand being affected with the current pandemic', 'What has the unemployment rate, due to the pandemic, done for our economic system and society How has the closers on the food industries, due to the pandemic, affected our economic system and society'], 2: ['is capitalism the most successful economic system', 'What country has been most successful with their economy and how did they go about it'], 3: ['How will you reach consensus on the definition of what is an equitable economic system']})
        self.assertEqual([int(x) for x in q_clus],[0,0,1,2,0,0,0,3,2,1])

        self.json_data2 = self.json_data[:6]
        embed2, data_used2, _ = get_data_embeddings(self.json_data2)

        tst_X2 = np.array([[ 0.0000000e+00  ,5.8276892e-01,  6.6471207e-01,  1.0000000e+00,
                            4.0161848e-01,  1.0000000e+00],
                            [ 5.8276892e-01, -1.1920929e-07 , 1.0000000e+00 , 6.0412532e-01,
                            5.3194344e-01 , 5.3522450e-01],
                            [ 6.6471207e-01,  1.0000000e+00 , 0.0000000e+00 , 1.0000000e+00,
                            5.8649641e-01 , 1.0000000e+00],
                            [ 1.0000000e+00 , 6.0412532e-01 , 1.0000000e+00 , 2.3841858e-07,
                            1.0000000e+00 , 1.0000000e+00],
                            [ 4.0161848e-01 , 5.3194344e-01 , 5.8649641e-01 , 1.0000000e+00,
                            1.1920929e-07 , 5.0259256e-01],
                            [ 1.0000000e+00 , 5.3522450e-01 , 1.0000000e+00 , 1.0000000e+00,
                            5.0259256e-01 , 0.0000000e+00]])
        grouped2, q_clus2, X2 = HAC_with_Sparsification(4, embed2, data_used2, 2, 1)
        self.assertEqual(X2.all(),tst_X2.all())   
        tst_grp2 = {0: ['How is social distancing going to impact the economy', 'How does the unemployment rate effect the economy'], 1: ['Does helping people out of poverty help the economy', 'If the federal government raised minimum wage, would it positively effect the national economy'], 2: ['How is supply and demand being affected with the current pandemic'], 3: ['is capitalism the most successful economic system']}
        tst_qcls2 = [0 ,1 ,2 ,3 ,0 ,1]
        self.assertEqual(grouped2,tst_grp2)
        self.assertEqual([int(x) for x in q_clus2],tst_qcls2)    

    """ Testing the Best HAC with Sparsification Function: Optimal Number Test + Correct Clustering
    """


    def test_best_HAC_sparse(self):

        self.json_data2 = self.json_data[:6]
        embed2, data_used2, _ = get_data_embeddings(self.json_data2)
        grp2, q_cls2 = best_score_HAC_sparse(embed2,data_used2,2)
        tst_grp2 = {0: ['How is social distancing going to impact the economy', 'How does the unemployment rate effect the economy'], 1: ['Does helping people out of poverty help the economy', 'If the federal government raised minimum wage, would it positively effect the national economy'], 2: ['How is supply and demand being affected with the current pandemic'], 3: ['is capitalism the most successful economic system']}
        tst_qcls2 = [0 ,1 ,2 ,3 ,0 ,1]
        self.assertEqual(grp2,tst_grp2)
        self.assertEqual([int(x) for x in q_cls2],tst_qcls2)    

        embed, data_used, _ = get_data_embeddings(self.json_data)
        grp, q_cls = best_score_HAC_sparse(embed,data_used,2)

        self.assertEqual(grp,{0: ['How is social distancing going to impact the economy', 'Does helping people out of poverty help the economy', 'How does the unemployment rate effect the economy', 'If the federal government raised minimum wage, would it positively effect the national economy', 'How would a free market system affect the private sector'], 1: ['How is supply and demand being affected with the current pandemic', 'What has the unemployment rate, due to the pandemic, done for our economic system and society How has the closers on the food industries, due to the pandemic, affected our economic system and society'], 2: ['is capitalism the most successful economic system', 'What country has been most successful with their economy and how did they go about it'], 3: ['How will you reach consensus on the definition of what is an equitable economic system']})
        self.assertEqual([int(x) for x in q_cls],[0,0,1,2,0,0,0,3,2,1])

    """ Testing the Best HAC Normal Function: Optimal Number Test + Correct Clustering
    """


    def test_best_HAC_normal(self):

        embed, data_used, _ = get_data_embeddings(self.json_data)
        grp, q_cls = get_best_HAC_normal(embed,data_used)
        self.assertEqual(grp,{0: ['Does helping people out of poverty help the economy', 'If the federal government raised minimum wage, would it positively effect the national economy', 'How would a free market system affect the private sector'], 1: ['is capitalism the most successful economic system', 'What country has been most successful with their economy and how did they go about it'], 2: ['How is social distancing going to impact the economy', 'How is supply and demand being affected with the current pandemic', 'How does the unemployment rate effect the economy', 'What has the unemployment rate, due to the pandemic, done for our economic system and society How has the closers on the food industries, due to the pandemic, affected our economic system and society'], 3: ['How will you reach consensus on the definition of what is an equitable economic system']})
        self.assertEqual([int(x) for x in q_cls],[2, 0, 2, 1, 2, 0, 0, 3 ,1 ,2])        

        self.json_data2 = self.json_data[:6]
        embed2, data_used2, _ = get_data_embeddings(self.json_data2)
        
        grp2, q_cls2 = get_best_HAC_normal(embed2,data_used2)

        self.assertEqual(grp2,{0: ['Does helping people out of poverty help the economy', 'If the federal government raised minimum wage, would it positively effect the national economy'], 1: ['How is social distancing going to impact the economy', 'How does the unemployment rate effect the economy'], 2: ['is capitalism the most successful economic system'], 3: ['How is supply and demand being affected with the current pandemic']})
        self.assertEqual([int(x) for x in q_cls2],[1, 0, 3, 2, 1, 0])

    """ Testing the Get Cluster Dictionary Function: Correct Format + Data Test
    """


    def test_cluster_dict(self):
        embed,data_used,q_ids = get_data_embeddings(self.json_data)
        grp, q_cls = best_score_HAC_sparse(embed,data_used,2)
        fin_cls = return_cluster_dict(q_cls, q_ids)
        self.assertEqual(fin_cls,{0: [2, 2, 2, 2, 2], 1: [2, 2], 2: [2, 2], 3: [2]})

        embed2,data_used2,q_ids2 = get_data_embeddings(self.json_data[:6])
        grp2, q_cls2 = best_score_HAC_sparse(embed2,data_used2,2)
        fin_cls2 = return_cluster_dict(q_cls2, q_ids2)
        self.assertEqual(fin_cls2,{0: [2, 2], 1: [2, 2], 2: [2], 3: [2]})
        


   

if __name__=='__main__':
    unittest.main()
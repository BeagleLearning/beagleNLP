from helper_functions import get_questions, display_clusters
from word_embedding_methods import WordEmbeddings
from transformer_embedding_methods import TransformerEmbeddings
from lsa import LatentSemanticAnalyis
from lda import LatentDirichletAllocation
from greedy_clustering import GreedyClustering
from Testing.testing import Test

test_class = Test()
#Word Embeddings
def word_embeddings(test = True, display = False):
    WE = WordEmbeddings()
    if test == False:
        questions = get_questions('normal')
        clusters = WE.fasttext(questions)
        if display:
            questions_untokenized = get_questions('normal',tokenized=False)
            display_clusters(questions_untokenized, clusters, 'WORD EMBEDDINGS WITH AGGLOMERATIVE CLUSTERING')
    else:
        args = ['pretrained']
        test_class.test_clustering_algorithm(WE.fasttext, arguments = args, display_clusters = display)

#LDA
def lda(test = True, display = False):
    LDA = LatentDirichletAllocation()
    if test == False:
        questions = get_questions('normal')
        clusters = LDA.lda(questions)
        if display:
            display_clusters(questions, clusters, 'LATENT DIRICHLET ALLOCATION')
    else:
        test_class.test_clustering_algorithm(LDA.lda, display_clusters = display)

#LSA
def lsa(test = True, display = False):
    LSA = LatentSemanticAnalyis()
    if test == False:
        questions = get_questions('normal',tokenized=False)
        clusters = LSA.lsa(questions)
        if display:
            display_clusters(questions, clusters, 'LATENT SEMANTIC ANALYSIS')
    else:
        test_class.test_clustering_algorithm(LSA.lsa, display_clusters = display)

#Transformer Embeddings
def transformer_embeddings(test = True, display = False):
    TE = TransformerEmbeddings()
    if test == False:
        questions = get_questions('normal',tokenized = False)
        clusters = TE.transformer_embeddings(questions)
        if display:
            display_clusters(questions,clusters,'TRANSFORMER EMBEDDINGS WITH AGGLOMERATIVE CLUSTERING')
    else:
        test_class.test_clustering_algorithm(TE.transformer_embeddings, display_clusters = display)

#Greedy Clustering
def greedy_clustering(test = True, display = False):
    GC = GreedyClustering()
    if test == False:
        questions = get_questions('normal')
        clusters = GC.greedy_clustering(questions)
        if display:
            display_clusters(questions,clusters,'GREEDY CLUSTERING')
    else:
        test_class.test_clustering_algorithm(GC.greedy_clustering, display_clusters = display)



#word_embeddings(test = True, display = False)
#lda(test = False )
#lsa(test = True, display = True)
#transformer_embeddings(test = False, display = True)
greedy_clustering(test = True, display = False)
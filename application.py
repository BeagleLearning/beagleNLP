# pylint: disable=E1101
""" Flask application for Beagle's NLP analysis """
import os
import logging
from flask import Flask, request, jsonify, send_file, g
import analysis
import numpy as np
from build_tag_cluster import buildTagCluster
from beagleError import BeagleError
import errors
from use_cluster import get_data_embeddings, best_score_HAC_sparse, HAC_with_Sparsification, get_best_HAC_normal, return_cluster_dict
import time
from functools import wraps

# Typing and deduplication imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import tensorflow_hub as hub
import json
from classifier_10_cats import get_predictions
from use_deduplication import deduplicate


if "PYTHON_ENVIRONMENT" in os.environ.keys() and os.environ['PYTHON_ENVIRONMENT'] == "production":
    logging.basicConfig(
        filename='/opt/python/log/application.log',
        level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
# some bits of text for the page.
HEADER_TEXT = '''
    <html>\n<head> <title>Beagle NLP API</title> </head>\n<body>'''
END_OF_PAGE = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__, static_url_path='/static/')
application.logger.info("Flask app created!")

"""
#For timing purposes
@application.before_request
def before_req_func():
    g.timings = {}

@application.after_request
def after_req_func(response):
    response.data += ('\n' + str(g.timings)).encode('ascii')
    return response



def time_this(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        g.timings[func.__name__] = end-start
        return r
    return wrapper

"""

@application.route("/", methods=["GET"])
def index_route():
    """ Confirms the app is running """
    return HEADER_TEXT + "Welcome to the Beagle NLP API" + END_OF_PAGE


@application.route("/playground", methods=["GET"])
def playground():
    """ Renders the playground page """
    return application.send_static_file("playground.html")


@application.route("/robots.txt", methods=["GET"])
def robots():
    """ Tells robots to behave themselves """
    return send_file("static/robots.txt")


@application.route("/word2vec/<token>", methods=["GET"])
def word2vec(token):
    """ Returns our analysis's vector representation of a token"""
    vector = analysis.getVector(token)
    return jsonify(vector.tolist())


@application.route("/cluster/", methods=["POST"])
def cluster_words():
    """ Clusters questions sent in request body """
    corpus = None
    data = request.get_json()
    if "questions" not in data:
        raise BeagleError(errors.MISSING_PARAMETERS_FOR_ROUTE)

    if len(data["questions"]) < 2:
        raise BeagleError(errors.TOO_FEW_QUESTIONS)

    if not "agglomerate" in data:
        data["agglomerate"] = False

    application.logger.info(f"Questions: {data['questions']}")
    if "keywords" in data and data['keywords'] is not None and len(data['keywords']) > 0:
        application.logger.info(f"Keywords found! {data['keywords']}")
        questions, keywords, agglomerate = data["questions"], data["keywords"], data["agglomerate"]
        corpus = analysis.clusterQuestionsOnKeywords(questions, keywords, agglomerate)
        corpus = jsonify(corpus)
    else:
        application.logger.info("No keywords found.")
        corpus = analysis.cluster_questions(data["questions"])
        corpus = jsonify(buildTagCluster(corpus))

    return corpus


@application.route("/cluster/categorize-new-questions", methods=["POST"])
def categorize_orphan_questions():
    """ Matches uncategorized questions with keywords ("clusters") sent in request body """
    data = request.get_json()
    for param in ["questions", "clusters"]:
        if param not in data or data[param] is None or len(data[param]) == 0:
            raise BeagleError(errors.MISSING_PARAMETERS_FOR_ROUTE)
    application.logger.info(f"Questions: {data['questions']}")
    application.logger.info(f"Keywords: {data['clusters']}")
    clusters = analysis.match_questions_with_categories(data["questions"], data["clusters"])
    return jsonify(clusters)


@application.route("/play/cluster/", methods=["POST"])
def playground_cluster_words():
    """ Experimental route for testing different clustering techniques """
    corpus = None
    data = request.get_json()
    if "questions" not in data:
        raise BeagleError(errors.MISSING_PARAMETERS_FOR_ROUTE)

    if len(data["questions"]) < 2:
        raise BeagleError(errors.TOO_FEW_QUESTIONS)

    algorithm = data["algorithm"]
    algorithm_params = data["algorithmParams"]
    remove_outliers = data["removeOutliers"]
    question_list = [
        {
            "id": qn["id"],
            "question": qn["question"].lower().strip()
        } for qn in data["questions"]]
    if "keywords" in data and data['keywords'] is not None:
        application.logger.info(f"Keywords found! {data['keywords']}")
        corpus = analysis.clusterQuestionsOnKeywords(
            question_list, data["keywords"], True)
    else:
        application.logger.info("No keywords found.")
        corpus = analysis.customClusterQuestions(question_list, algorithm, algorithm_params,
                                                 remove_outliers)

    return jsonify(buildTagCluster(corpus))


@application.errorhandler(BeagleError)
def handle_invalid_usage(error):
    """ Simple Flask app error handling """
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

"""CUSTOM ROUTE: Sparse HAC"""
@application.route("/useclustersparse/", methods=["POST"])
def handleUSECluster():
    
    data = request.get_json()
    application.logger.debug(type(data)) #List
    application.logger.debug(len(data))
    if(len(data) < 6):
        raise BeagleError(errors.TOO_FEW_QUESTIONS)
    
    
    embeddings, data_used_for_demo, q_ids_list = get_data_embeddings(data)
    best_scores = list(map(int,best_score_HAC_sparse(embeddings, data_used_for_demo, 2)[1]))
    return jsonify(return_cluster_dict(best_scores,q_ids_list))
    

"""CUSTOM ROUTE 2: Normal HAC"""
@application.route("/useclusternormal/", methods=["POST"])
def handleUSECluster2():
    
    data = request.get_json()
    application.logger.debug((type(data))) #List
    application.logger.debug(len(data))
    if(len(data) < 6):
        raise BeagleError(errors.TOO_FEW_QUESTIONS)
    
    embeddings, data_used_for_demo, q_ids_list = get_data_embeddings(data)
    
    best_scores = list(map(int,get_best_HAC_normal(embeddings, data_used_for_demo)[1]))
    return jsonify(return_cluster_dict(best_scores,q_ids_list))
    
    

    
"""CUSTOM ROUTE 3: Condition Based HAC"""
@application.route("/usecondition/", methods=["POST"])

def handleUSECluster3():
    
    
    data = request.get_json()
    application.logger.debug((type(data))) #List
    application.logger.debug(len(data))
    if(len(data) < 6):
        raise BeagleError(errors.TOO_FEW_QUESTIONS) #Communicate we don't support
    
    embeddings, data_used_for_demo, q_ids_list = get_data_embeddings(data)
    if(len(data_used_for_demo)<50):
        
        best_scores = list(map(int,best_score_HAC_sparse(embeddings, data_used_for_demo, 2)[1]))
        return jsonify(return_cluster_dict(best_scores,q_ids_list))
    
        
    else:
        
        best_scores = list(map(int,get_best_HAC_normal(embeddings, data_used_for_demo)[1]))
        return jsonify(return_cluster_dict(best_scores,q_ids_list))
    

##### TYPING AND DEDUPLICATION ROUTES #####

@application.route("/type/", methods=['POST'])
def classify_question_types():
    data = request.get_json()
    if "questions" not in data:
        raise BeagleError(errors.MISSING_PARAMETERS_FOR_ROUTE)

    categorized_questions = get_predictions(data['questions'], device = device,\
            tokenizer = tokenizer, model=model)

    return jsonify(categorized_questions)


@application.route("/deduplicate/", methods=['POST'])
def deduplicate_questions():
    data = request.get_json()
    if "questions" not in data:
        raise BeagleError(errors.MISSING_PARAMETERS_FOR_ROUTE)

    grouped_duplicates = deduplicate(data['questions'], embedder = use_embedder)

    return jsonify(grouped_duplicates)


# run the app.
if __name__ == "__main__":

    ##### DEBERTA INITIATION AND TORCH DEVICE MOUNT #####
    model_location = './resources/10_cats_deberta/torch_hf_deberta_epoch_5.model'

    try:
        # declare device for torch to mount
        device = torch.device('cpu') #'cuda' if torch.cuda.is_available()
        # configuration necessary for the right initiation of the model
        config = AutoConfig.from_pretrained("microsoft/deberta-base-mnli")
        config.num_labels = 10
        # pretrained model initiation code
        model = AutoModelForSequenceClassification.from_config(config)
        model.to(device)
        model.load_state_dict(torch.load(model_location, map_location=torch.device('cpu'))) #TODO: remove hardcoded string if CUDA available
        # corresponding tokenizer initiation
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base-mnli")
    except:
        raise BeagleError(errors.DEBERTA_LOAD_ERROR) #If model not loaded

    ##### ##### ##### #####

    ##### UNIVERSAL SENTENCE ENCODER INITIATION #####
    use_location = './resources/use_4'
    try:
        use_embedder = hub.load(use_location)
    except:
        raise BeagleError(errors.USE_LOAD_ERROR) #If model not loaded
    ##### ##### ##### #####

    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.run()

from flask import Flask, request, jsonify, send_file
from beagleError import BeagleError
import errors
#import analysis
import logging
#from build_tag_cluster import buildTagCluster
#import textrank
import time
import os

if "PYTHON_ENVIRONMENT" in os.environ.keys() and os.environ['PYTHON_ENVIRONMENT'] == "production":
    logging.basicConfig(
        filename='/opt/python/log/application.log',
        level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
# some bits of text for the page.
headerText = '''
    <html>\n<head> <title>Beagle NLP API</title> </head>\n<body>'''
endOfPage = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__, static_url_path='/static/')

application.logger.info("Flask app created!")


@application.route("/", methods=["GET"])
def indexRoute():
    return headerText + "Welcome to the Beagle NLP API" + endOfPage


@application.route("/playground", methods=["GET"])
def playground():
    # return render_template('playground.html')
    return application.send_static_file("playground.html")


@application.route("/robots.txt", methods=["GET"])
def robots():
    return send_file("static/robots.txt")


"""
@application.route("/word2vec/<token>", methods=["GET"])
def word2vec(token):
    vector = analysis.getVector(token)
    return jsonify(vector.tolist())


@application.route("/cluster/", methods=["POST"])
def clusterWords():
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
        corpus = analysis.clusterQuestionsOnKeywords(data["questions"], data["keywords"], data["agglomerate"])
        corpus = jsonify(corpus)
    else:
        application.logger.info("No keywords found.")
        corpus = analysis.clusterQuestions(data["questions"])
        corpus = jsonify(buildTagCluster(corpus))

    return corpus


@application.route("/cluster/categorize-new-questions", methods=["POST"])
def categorizeOrphanQuestions():
    corpus = None
    data = request.get_json()
    if "clusters" not in data or len(data["clusters"]) == 0:
        raise BeagleError(errors.MISSING_PARAMETERS_FOR_ROUTE)

    if data["questions"] is None or len(data["questions"]) == 0:
        raise BeagleError(errors.MISSING_PARAMETERS_FOR_ROUTE)

    application.logger.info(f"Questions: {data['questions']}")
    corpus = analysis.matchQuestionsWithCategories(data["questions"], data["clusters"])

    return jsonify({})


@application.route("/play/cluster/", methods=["POST"])
def playgroundClusterWords():
    corpus = None
    data = request.get_json()
    if "questions" not in data:
        raise BeagleError(errors.MISSING_PARAMETERS_FOR_ROUTE)

    if len(data["questions"]) < 2:
        raise BeagleError(errors.TOO_FEW_QUESTIONS)

    algorithm = data["algorithm"]
    algorithmParams = data["algorithmParams"]
    removeOutliers = data["removeOutliers"]
    question_list = [
        {
            "id": qn["id"],
            "question": qn["question"].lower().strip()
        } for qn in data["questions"]]
    if "keywords" in data and data['keywords'] is not None:
        application.logger.info(f"Keywords found! {data['keywords']}")
        corpus = analysis.clusterQuestionsOnKeywords(
            question_list, data["keywords"])
    else:
        application.logger.info("No keywords found.")
        corpus = analysis.customClusterQuestions(question_list, algorithm, algorithmParams, removeOutliers)

    return jsonify(buildTagCluster(corpus))
"""

@application.errorhandler(BeagleError)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.run()

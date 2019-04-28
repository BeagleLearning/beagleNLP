from flask import Flask, request, jsonify, send_file
from beagleError import BeagleError
import errors
import analysis
import logging
from build_tag_cluster import buildTagCluster

from tfidf import calc_keywords

# logging.basicConfig(
#     filename='/opt/python/log/application.log',
#     level=logging.DEBUG,
#     format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
# some bits of text for the page.
headerText = '''
    <html>\n<head> <title>Beagle NLP API</title> </head>\n<body>'''
endOfPage = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__)

application.logger.info("Flask app created!")


@application.route("/", methods=["GET"])
def indexRoute():
    return headerText + "Welcome to the Beagle NLP API" + endOfPage

@application.route("/playground", methods=["GET"])
def playground():
    return send_file("static/playground.html")

@application.route("/robots.txt", methods=["GET"])
def robots():
    return send_file("static/robots.txt")


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

    question_list = [{"id": qn["id"], "question": qn["question"].lower()} for qn in data["questions"]]
    application.logger.info(f"Questions: {data['questions']}")
    if "keywords" in data and data['keywords'] is not None:
        application.logger.info(f"Keywords found! {data['keywords']}")
        corpus = analysis.clusterQuestionsOnKeywords(
            question_list, data["keywords"])
    else:
        application.logger.info("No keywords found.")
        corpus = analysis.clusterQuestions(question_list)
        print("corpus.clusters.item")
        print(corpus.clusters)
        allquestions = [q["question"] for q in question_list]
        for key, questionsList in corpus.clusters.items():
            print(calc_keywords([q.text for q in questionsList], allquestions))
            # print()

        print("\n")
    return jsonify(buildTagCluster(corpus))


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

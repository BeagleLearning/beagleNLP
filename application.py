from flask import Flask, request, jsonify, send_file
from beagleError import BeagleError
import analysis

# some bits of text for the page.
headerText = '''
    <html>\n<head> <title>Beagle NLP API</title> </head>\n<body>'''
endOfPage = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__)

application.logger.info("Flask app created!")


@application.route("/", methods=["GET"])
def indexRoute():
    application.logger.info("Index route accessed.")
    return headerText + "Welcome to the Beagle NLP API" + endOfPage


@application.route("/robots.txt", methods=["GET"])
def robots():
    print("ACCESSING")
    return send_file("static/robots.txt")


@application.route("/word2vec/<token>", methods=["GET"])
def word2vec(token):
    vector = analysis.getVector(token)
    return jsonify(vector.tolist())


@application.route("/cluster/", methods=["POST"])
def clusterWords():
    body = request.data
    corpus = None
    if body.keywords:
        corpus = analysis.clusterQuestionsOnKeywords(
            body.questions, body.keywords)
    else:
        corpus = analysis.clusterQuestions(body.questions)
    return jsonify(corpus.clusters)


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

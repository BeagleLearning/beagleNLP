from flask import Flask, request, jsonify
import errors
from beagleError import BeagleError
import spacy
import analysis

nlp = spacy.load("./model/googlenews-spacy")

# some bits of text for the page.
headerText = '''
    <html>\n<head> <title>Beagle NLP API</title> </head>\n<body>'''
endOfPage = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__)


@application.route("/")
def indexRoute():
    return headerText + "Welcome to the Beagle NLP API" + endOfPage


@application.route("/word2vec/<token>", methods=["GET"])
def word2vec(token):
    nlpTokens = nlp(token)
    if len(nlpTokens) > 0:
        return jsonify(nlpTokens[0].vector.tolist())
    else:
        raise BeagleError(errors.NO_TOKEN_FOUND)


@application.route("/cluster/", methods=["POST"])
def clusterWords():
    body = request.data
    clusters = None
    if body.keywords:
        clusters = analysis.clusterQuestionsOnKeywords(
            body.questions, body.keywords)
    else:
        clusters = analysis.clusterQuestions(body.questions)
    return jsonify(clusters)


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

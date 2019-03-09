#import analysis as a

SAMPLE_DOCS = ["hello world", "hi world", "nothing similar", "hey globe",
               "hello globe", "not similar", "not same", "antagonistic yoda"]
BIRD_QUESTIONS = [
    {
        "question": "What is a roadrunner?",
        "id": 1
    }, {
        "question": "What is a coyote?",
        "id": 2
    }, {
        "question": "Do coyote eat roadrunners?",
        "id": 3
    }, {
        "question": "How many roadrunners do coyotes consume?",
        "id": 4
    }, {
        "question": "What is a roadruner?",
        "id": 5
    }, {
        "question": "How many birds do coyotes eat?",
        "id": 6
    }, {
        "question": "How many roadrunners does a coyote eat?",
        "id": 7
    }
]

#corp = a.clusterQuestionsOnKeywords(BIRD_QUESTIONS,
#                                    ["roadrunner", "coyote", "eat"])

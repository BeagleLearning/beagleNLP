import unittest
import requests
import json
import sys
sys.path.insert(0,'..')
import errors

class TestDuplicateGroupingRoute(unittest.TestCase):

    route = "http://localhost:5000/deduplicate/all/"

    def test_empty_list(self):
        data = []
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.TOO_FEW_QUESTIONS_TO_DEDUPLICATE)
        self.assertEqual(res.status_code, 400)

    def test_not_a_list(self):
        data = "test_string"
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.INVALID_INPUT_NOT_A_LIST)
        self.assertEqual(res.status_code, 500)

    def test_element_not_a_dict(self):
        data = [
            [],
                {
                "id": 2,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 3,
                "content": "When exactly did that happen?",
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.INVALID_FORMATTING_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_single_question(self):
        data = [
                {
                "id": 2,
                "content": "Why would anyone do that, though?",
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.TOO_FEW_QUESTIONS_TO_DEDUPLICATE)
        self.assertEqual(res.status_code, 400)

    def test_empty_values_1(self):
        data = [
                {
                "id": None,
                "content": "Why would anyone do that, though?",
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.TOO_FEW_QUESTIONS_TO_DEDUPLICATE)
        self.assertEqual(res.status_code, 400)

    def test_empty_values_2(self):
        data = [
                {
                "id": 1,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 3,
                "content": None,
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.EMPTY_VALUE_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_values_not_strings(self):
        data = [
                {
                "id": 1,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": [1,2,3],
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.UNEXPECTED_DATA_TYPE_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_invalid_formatting(self):
        data = [
                {
                "ids": 1,
                "contents": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": [1,2,3],
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.INVALID_FORMATTING_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_no_questions_key(self):
        data = [
                {
                "ids": 1,
                "contents": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": [1,2,3],
            },
        ]
        res = requests.post(url=self.route, json=data)
        self.assertEqual(res.json()['code'], errors.MISSING_PARAMETERS_FOR_ROUTE)
        self.assertEqual(res.status_code, 404)

    def test_id_a_string(self):
        data = [
                {
                "id": "1",
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": [1,2,3],
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.UNEXPECTED_DATA_TYPE_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_valid_input(self):
        data = [
            {
                "id": 1,
                "content": "What is the meaning of all this?",
            },
                {
                "id": 2,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 3,
                "content": "When exactly did that happen?",
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json(), [[0], [1], [2]])
        self.assertEqual(res.status_code, 200)

    def test_wrong_data_types(self):
        data = [
                {
                "id": 1,
                "content": 2,
            },
                {
                "id": 3,
                "content": 4,
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.UNEXPECTED_DATA_TYPE_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_questions_in_bulk(self):
        with open('./thousand_questions_test.json', 'r') as f:
            data = json.load(f)

        res = requests.post(url=self.route, json=data)
        self.assertEqual(len(res.json()), 722) # with threshold 0.7 only!
        self.assertEqual(res.status_code, 200)

    def test_empty_string(self):
        data = [
                {
                "id": 1,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": "",
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(len(res.json()), 2)
        self.assertEqual(res.status_code, 200)



class TestDuplicateDetectionRoute(unittest.TestCase):

    route = "http://localhost:5000/deduplicate/compare-one/"

    def test_target_empty_list(self):
        data = [
                {
                "id": 1,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": "When exactly did that happen?",
            },
        ]
        json_to_send = {"questions": data, "target": []}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.INVALID_FORMATTING_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_target_empty_dict(self):
        data = [
                {
                "id": 1,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": "When exactly did that happen?",
            },
        ]
        json_to_send = {"questions": data, "target": {}}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.INVALID_FORMATTING_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_target_empty_string(self):
        data = [
                {
                "id": 1,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": "When exactly did that happen?",
            },
        ]
        target = {
            "id": 42,
            "content": "",
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.INVALID_QUESTION_EMPTY_STRING_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_not_a_list(self):
        data = "test_string"
        target = {
            "id": 42,
            "content": "How is the weather today?",
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.INVALID_INPUT_NOT_A_LIST)
        self.assertEqual(res.status_code, 500)
    
    def test_no_target_key_in_json(self):
        data = "test_string"
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], "MISSING_PARAMETERS_FOR_ROUTE")
        self.assertEqual(res.status_code, 404)

    def test_element_not_a_dict(self):
        data = [
            [],
                {
                "id": 2,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 3,
                "content": "When exactly did that happen?",
            },
        ]
        target = {
            "id": 42,
            "content": "How is the weather today?",
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.INVALID_FORMATTING_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_single_question(self):
        data = [
                {
                "id": 2,
                "content": "Why would anyone do that, though?",
            },
        ]
        target = {
            "id": 42,
            "content": "But why would anyone do that?",
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json(), [2])
        self.assertEqual(res.status_code, 200)

    def test_empty_values_1(self):
        data = [
                {
                "id": None,
                "content": "Why would anyone do that, though?",
            },
        ]
        target = {
            "id": 42,
            "content": "But why would anyone do that?",
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.EMPTY_VALUE_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_empty_values_2(self):
        data = [
                {
                "id": 1,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 3,
                "content": None,
            },
        ]
        target = {
            "id": 42,
            "content": "But why would anyone do that?",
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.EMPTY_VALUE_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_values_not_strings(self):
        data = [
                {
                "id": 1,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": [1,2,3],
            },
        ]
        target = {
            "id": 42,
            "content": 4242,
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.UNEXPECTED_DATA_TYPE_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_invalid_formatting(self):
        data = [
                {
                "id": 1,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": "Why would anyone do that?",
            },
        ]
        target = {
            "ids": 42,
            "contents": 4242,
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.INVALID_FORMATTING_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_no_questions_key(self):
        data = [
                {
                "ids": 1,
                "contents": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": [1,2,3],
            },
        ]
        target = {
            "id": 42,
            "content": "But why would anyone do that?",
        }
        json_to_send = {"preguntas": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.MISSING_PARAMETERS_FOR_ROUTE)
        self.assertEqual(res.status_code, 404)

    def test_id_a_string(self):
        data = [
                {
                "id": "1",
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": [1,2,3],
            },
        ]
        target = {
            "id": 42,
            "content": "But why would anyone do that?",
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.UNEXPECTED_DATA_TYPE_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_valid_input(self):
        data = [
            {
                "id": 3,
                "content": "What is the meaning of all this?",
            },
                {
                "id": 1,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 2,
                "content": "When exactly did that happen?",
            },
        ]
        target = {
            "id": 42,
            "content": "But why would anyone do that?",
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json(), [1])
        self.assertEqual(res.status_code, 200)

    def test_wrong_data_types(self):
        data = [
                {
                "id": 1,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 3,
                "content": "What is the meaning of all this?",
            },
        ]
        target = {
            "id": 42,
            "content": 33,
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], errors.UNEXPECTED_DATA_TYPE_ERROR)
        self.assertEqual(res.status_code, 500)

    def test_questions_in_bulk(self):
        with open('./thousand_questions_test.json', 'r') as f:
            data = json.load(f)
        target = {
            "id": 42,
            "content": "What is the effect of covid-19 on the economy?",
        }
        json_to_send = {"questions": data["questions"], "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(len(res.json()), 5) # with threshold 0.7 only!
        self.assertEqual(res.status_code, 200)

    def test_empty_string(self):
        data = [
                {
                "id": 36,
                "content": "Why would anyone do that, though?",
            },
                {
                "id": 21,
                "content": "",
            },
        ]
        target = {
            "id": 42,
            "content": "But why would anyone do that?",
        }
        json_to_send = {"questions": data, "target": target}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json(), [36])
        self.assertEqual(res.status_code, 200)



if __name__ == "__main__":
    unittest.main()
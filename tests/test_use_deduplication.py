import unittest
import requests
import json
import sys
sys.path.insert(0,'..')
import errors

class TestDeduplicationRoute(unittest.TestCase):

    route = "http://localhost:5000/deduplicate/"

    def test_empty_list(self):
        data = []
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], 2813)

    def test_not_a_list(self):
        data = "test_string"
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], 2812)

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
        self.assertEqual(res.json()['code'], 2814)

    def test_single_question(self):
        data = [
                {
                "id": 2,
                "content": "Why would anyone do that, though?",
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], 2810)

    def test_empty_values_1(self):
        data = [
                {
                "id": None,
                "content": "Why would anyone do that, though?",
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], 2810)

    def test_empty_values_1(self):
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
        self.assertEqual(res.json()['code'], 2815)

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
        self.assertEqual(res.json()['code'], 2816)

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
        self.assertEqual(res.json()['code'], 2814)

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
        self.assertEqual(res.json()['code'], 2816)

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
        self.assertEqual(res.json()['code'], 2816)

    def test_questions_in_bulk(self):
        with open('./thousand_questions_test.json', 'r') as f:
            data = json.load(f)

        res = requests.post(url=self.route, json=data)
        self.assertEqual(len(res.json()), 722) # with threshold 0.7 only!


if __name__ == "__main__":
    unittest.main()
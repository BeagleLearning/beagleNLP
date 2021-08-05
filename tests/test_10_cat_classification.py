import unittest
import requests
import json
import sys
sys.path.insert(0,'..')
import errors

class Test10CatsRoute(unittest.TestCase):

    """
    This testing suite, when checking for the right results, looks rather
    for the right structure and data types than explicit right results
    since the predictions might differ between multiple prediction runs.
    """

    route = "http://localhost:5000/type/"

    def test_empty_list(self):
        data = []
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], 2813)
        self.assertEqual(res.status_code, 500)

    def test_not_a_list(self):
        data = "test_string"
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], 2812)
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
        self.assertEqual(res.json()['code'], 2814)
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
        self.assertEqual(list(res.json()[0].keys()), ['id','type'])
        self.assertEqual(type(res.json()[0]['id']), int)
        self.assertEqual(res.json()[0]['id'], 2)
        self.assertEqual(type(res.json()[0]['type']), int)
        self.assertEqual(res.status_code, 200)
    

    def test_empty_values_1(self):
        data = [
                {
                "id": None,
                "content": "Why would anyone do that, though?",
            },
        ]
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], 2815)
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
        json_to_send = {"questions": data}
        res = requests.post(url=self.route, json=json_to_send)
        self.assertEqual(res.json()['code'], 2815)
        self.assertEqual(res.status_code, 500)

    def test_contents_not_strings(self):
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
        self.assertEqual(res.json()['code'], 2814)
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
        self.assertEqual(res.json()['code'], 2816)
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
        self.assertEqual(len(res.json()), 3)
        for result_dict in res.json():
            self.assertEqual(type(result_dict['id']), int)
            self.assertEqual(type(result_dict['type']), int)

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
        self.assertEqual(res.json()['code'], 2816)
        self.assertEqual(res.status_code, 500)

    def test_questions_in_bulk(self):
        with open('./thousand_questions_test.json', 'r') as f:
            data = json.load(f)

        res = requests.post(url=self.route, json=data)
        self.assertEqual(len(res.json()), 1000)
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
        self.assertIsNone(res.json()[1])
        self.assertEqual(res.status_code, 200)
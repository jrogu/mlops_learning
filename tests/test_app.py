import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from app import app, classify_image, get_labels
from create_db import add_row, connect
from PIL import Image
import io

class TestFlaskApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        
    # TODO
import unittest

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..\..')))

from src.sentimentClassifier.data.preprocess.text_cleaner import ResourceLoader, TextCleaner

class TestResourceLoader(unittest.TestCase):
    
    def test_load_valid_files(self):
        # Paths to your test resource files
        test_emojis_path = 'data/resources/emojis_hun.txt'
        test_stopwords_path = 'data/resources/stopwords.txt'
        test_punctuations_path = 'data/resources/accepted_punctuations.txt'

        # Create a ResourceLoader instance with test files
        loader = ResourceLoader(test_emojis_path, test_stopwords_path, test_punctuations_path)

        # Assertions to ensure the files are loaded correctly
        self.assertIsInstance(loader.emojis, dict, "Emojis should be a dictionary")
        self.assertIsInstance(loader.stopwords, set, "Stopwords should be a set")
        self.assertIsInstance(loader.punctuations, set, "Punctuations should be a set")
       

class TestTextCleaner(unittest.TestCase):
    def setUp(self):
        # Setup with test data
        emojis_path = 'data/resources/emojis_hun.txt'
        stopwords_path = 'data/resources/stopwords.txt'
        punctuations_path = 'data/resources/accepted_punctuations.txt'
        self.loader = ResourceLoader(emojis_path, stopwords_path, punctuations_path)
        self.cleaner = TextCleaner(loader=self.loader)

    def test_remove_username(self):
        self.assertEqual(self.cleaner._remove_username("@user Ez egy teszt"), " Ez egy teszt")

    def test_remove_emails(self):
        self.assertEqual(self.cleaner._remove_emails("test@example.com az emailem"), " az emailem")

    def test_remove_urls(self):
        self.assertEqual(self.cleaner._remove_urls("Látogasd meg https://example.com"), "Látogasd meg ")

    def test_remove_html_tags(self):
        self.assertEqual(self.cleaner._remove_html_tags("<p>Ez egy teszt</p>"), "Ez egy teszt")

    def test_remove_special_chars(self):
        # Assuming '!' is not in your accepted punctuations
        self.assertEqual(self.cleaner._remove_special_chars("Hello$ Ez egy teszt $"), "Hello  Ez  egy  teszt  ")

    def test_remove_multiple_accepted_punct(self):
        # Assuming '.' is in your accepted punctuations
        self.assertEqual(self.cleaner._remove_multiple_accepted_punct("Ez egy teszt..."), "Ez egy teszt.")

    def test_remove_digits(self):
        self.assertEqual(self.cleaner._remove_digits("123 Ez egy teszt 456"), " Ez egy teszt ")

    def test_remove_stopwords(self):
        # Assuming 'this' and 'is' are in your stopwords
        self.assertEqual(self.cleaner._remove_stopwords("Ez egy teszt"), "teszt")

    def test_replace_emojis(self):
        # Assuming you have an emoji ':-)' mapped to 'SMILE' in your test emojis
        self.assertEqual(self.cleaner._replace_emojis("Hello :-)"), "Hello  mosoly ")

    def test_process(self):
        texts = ["@user Ez$ egy teszt! 123 :-)", "Minden okés <h1>volt tegnap</h1>?"]
        expected_output = ["teszt ! mosoly", "okés tegnap ?"]  # Expected after all cleaning steps
        self.assertEqual(self.cleaner.process(texts), expected_output)
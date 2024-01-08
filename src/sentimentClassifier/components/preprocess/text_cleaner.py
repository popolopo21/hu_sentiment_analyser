import re
from bs4 import BeautifulSoup
from string import punctuation, digits
import logging
from .resource_loader import ResourceLoader
from typing import List
from tqdm import tqdm

class TextCleaner():
     
    def __init__(self, loader: ResourceLoader, replace_emojis=True, remove_username=True, remove_emails=True,
                 remove_urls=True, remove_html_tags=True, remove_special_chars=True,
                 remove_multiple_accepted_punct=True, remove_digits=True, remove_stopwords=True):
        self.loader = loader
        self.replace_emojis = replace_emojis
        self.remove_username = remove_username
        self.remove_emails = remove_emails
        self.remove_urls = remove_urls
        self.remove_html_tags = remove_html_tags
        self.remove_special_chars = remove_special_chars
        self.remove_multiple_accepted_punct = remove_multiple_accepted_punct
        self.remove_digits = remove_digits
        self.remove_stopwords = remove_stopwords
        self.unwanted_chars = ''.join(set(punctuation) - loader.accepted_punctuation)

    def _remove_username(self, text: str)-> str:
        return re.sub(r'@\w+', '', text)

    def _remove_emails(self, text: str) -> str:
        return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', '', text)

    def _remove_urls(self, text: str) -> str:
        pattern = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
        return re.sub(pattern, '', text)

    def _remove_html_tags(self, text: str) -> str:
        return BeautifulSoup(text, 'lxml').get_text().strip()

    def _remove_special_chars(self, text: str) -> str:
        text = text.translate(str.maketrans('', '', self.unwanted_chars))
        for c in self.loader.accepted_punctuation:
            text = text.replace(c, f' {c}')  # add a space before accepted punctuations
        return text

    def _remove_multiple_accepted_punct(self, text: str) -> str:
        for c in self.loader.accepted_punctuation:
            text = re.sub(rf'\{c}+', c, text)
        return text

    def _remove_digits(self, text: str) -> str:
        return text.translate(str.maketrans('', '', digits))

    def _remove_stopwords(self, text: str) -> str:
        return ' '.join([t for t in text.split() if t.lower() not in self.loader.stopwords])

    def _replace_emojis(self, text: str) -> str:
        for em, word in self.loader.emojis.items():
            text = text.replace(em, ' ' + word + ' ')
        return text

    def process(self, texts: List[str]) -> List[str]:
        if texts is None:
            logging.warning("No texts provided for cleaning.")
            return []
        cleaned_texts = []
        for text in tqdm(texts, desc = "Processing texts"):
            try:
                if self.replace_emojis:
                    text = self._replace_emojis(text)

                if self.remove_username:
                    text = self._remove_username(text)

                if self.remove_emails:
                    text = self._remove_emails(text)

                if self.remove_urls:
                    text = self._remove_urls(text)

                if self.remove_html_tags:
                    text = self._remove_html_tags(text)

                if self.remove_multiple_accepted_punct:
                    text = self._remove_multiple_accepted_punct(text)

                if self.remove_special_chars:
                    text = self._remove_special_chars(text)

                if self.remove_digits:
                    text = self._remove_digits(text)

                if self.remove_stopwords:
                    text = self._remove_stopwords(text)

                cleaned_texts.append(text)

            except Exception as e:
                logging.error(f"Error processing text: {text}. Error: {e}")

        return cleaned_texts
    
    def process_in_batches(self, texts: List[str], batch_size: int) -> List[str]:
        cleaned_texts = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            cleaned_batch = self.process(batch)
            cleaned_texts.extend(cleaned_batch)
        return cleaned_texts

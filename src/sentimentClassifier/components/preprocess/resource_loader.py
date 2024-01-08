class ResourceLoader:
    """Class to load and store external resources like stopwords, emojis, and punctuations."""
    def __init__(self, emojis_path=None, stopwords_path=None, accepted_punctuation_path=None):
        self.emojis = self.load_emojis(emojis_path)
        self.stopwords = self.load_stopwords(stopwords_path)
        self.accepted_punctuation = self.load_accepted_punctuations(accepted_punctuation_path)
        
    def load_accepted_punctuations(self, path):
        accepted_punct = set()
        with open(path, 'r', encoding='utf-8') as fp:
            line = fp.read()
        for c in line:
            accepted_punct.add(c)

        return accepted_punct

    def load_emojis(self, path):
        emojis = {}
        with open(path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        for line in lines:
            em = line.rsplit('-', 1)
            icon = em[0][:-1]
            meaning = em[1][1:].strip()
            emojis[icon] = meaning

        return emojis

    def load_stopwords(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            stopwords = f.readlines()
        stopwords = set([x.strip() for x in stopwords])

        return stopwords
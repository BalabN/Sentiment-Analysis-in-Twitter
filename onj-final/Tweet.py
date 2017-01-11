from nltk import word_tokenize, PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class Tweet(object):
    lemmatizer = ""
    stemmer = ""

    sentiment = 0
    message = ""
    lemmatized = []
    tokens = []
    stemmed = []
    english_stops = []

    def __init__(self, sentiment, message):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.english_stops = set(stopwords.words('english'))

        self.sentiment = sentiment
        self.message = message
        self.create_tokens()
        self.lemmatize_tokens()
        self.stem_tokens()

    def lemmatize_tokens(self):
        self.remove_stopwords()
        self.lemmatized = [self.lemmatizer.lemmatize(token) for token in self.tokens]

    def stem_tokens(self):
        self.remove_stopwords()
        self.stemmed = [self.stemmer.stem(token) for token in self.tokens]

    def create_tokens(self):
        self.tokens = word_tokenize(self.message, "English")

    def remove_stopwords(self):
        return [word for word in self.tokens if word not in self.english_stops]

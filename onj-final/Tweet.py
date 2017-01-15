import string
import re

import numpy as np
import pandas as pd
from nltk import word_tokenize, PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

from ProfanityFilter import ProfanityFilter


class Tweet(object):
    POSITIVE = 1
    NEUTRAL = 2
    NEGATIVE = 3
    sentiments = [POSITIVE, NEGATIVE, NEUTRAL]
    URL_REGEX = '[hH][tT][tT][pP][sS]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    lemmatizer = ""
    stemmer = ""

    id = 0
    sentiment = 0
    message = ""
    message_without_punctuation = ""
    message_without_url = ""
    profanity_count = 0
    lemmatized = []
    tokens = []
    stemmed = []
    english_stops = []

    def __init__(self, tId, sentiment, message):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.english_stops = set(stopwords.words('english'))

        self.id = tId
        self.get_sentiment(sentiment)
        self.message = message.lower()
        self.profanity_count = self.get_profanity_count()
        self.message_without_url = self.remove_url()
        self.message_without_punctuation = self.remove_punctuation()
        self.create_tokens()
        self.replace_synonyms()
        self.lemmatize_tokens()
        self.stem_tokens()

    def __init__(self, tId, topic, sentiment, message):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.english_stops = set(stopwords.words('english'))

        self.id = tId
        self.get_sentiment(sentiment)
        self.message = message
        self.message_without_url = self.remove_url()
        self.message_without_punctuation = self.remove_punctuation(True)
        self.create_tokens()
        self.replace_synonyms()
        self.lemmatize_tokens()
        self.stem_tokens()

    def get_sentiment(self, sentiment):
        if sentiment == 'positive':
            self.sentiment = self.POSITIVE
        elif sentiment == 'neutral':
            self.sentiment = self.NEUTRAL
        else:
            self.sentiment = self.NEGATIVE

    def lemmatize_tokens(self):
        self.lemmatized = [self.lemmatizer.lemmatize(token) for token in self.remove_stopwords()]

    def stem_tokens(self):
        self.stemmed = [self.stemmer.stem(token) for token in self.remove_stopwords()]

    def get_profanity_count(self):
        filter = ProfanityFilter()
        return filter.count_bad_words(self.message)

    def create_tokens(self):
        self.tokens = word_tokenize(self.message_without_punctuation, "English")

    def remove_stopwords(self):
        return [word for word in self.tokens if word not in self.english_stops]

    def remove_punctuation(self, with_sentiment):
        m = pd.read_csv('data/emoticons.tsv', sep='\t').as_matrix()
        for n in m:
            if with_sentiment:
                self.message_without_url = self.message_without_url.replace(n[0], n[2])
            else:
                self.message_without_url = self.message_without_url.replace(n[0], "emotikon")
        table = self.message_without_url.maketrans({key: None for key in string.punctuation})
        return self.message_without_url.translate(table)

    def remove_url(self):
        return re.sub(self.URL_REGEX, '', self.message)

    def replace_synonyms(self):
        for i in range(len(self.tokens)):
            for synset in wordnet.synsets(self.tokens[i]):
                if len(synset.lemmas()) > 0:
                    self.tokens[i] = synset.lemmas()[0].name()

    def get_bayes_data(self, fdist):
        data = []
        print()
        for token in self.lemmatized:
            if fdist[0][0].contains(token):
                data.append("")

    @staticmethod
    def get_all_messages(tweets):
        messages = []
        for tweet in tweets:
            messages.append(tweet.message_without_punctuation)
        return np.array(messages)

    @staticmethod
    def get_all_messages_sentiment(tweets, sentiment):
        messages = []
        for tweet in tweets:
            if tweet.sentiment == sentiment:
                messages.append(tweet.message_without_punctuation)
        return np.array(messages)

    @staticmethod
    def get_all_sentiment(tweets):
        sentiments = []
        for tweet in tweets:
            sentiments.append(tweet.sentiment)
        return np.array(sentiments)

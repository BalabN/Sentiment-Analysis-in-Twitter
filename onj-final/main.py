import nltk
from nltk import FreqDist

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from PlotUtils import show_bar_plot
from ProfanityFilter import ProfanityFilter
from Tweet import Tweet
from sklearn.ensemble import RandomForestClassifier
import Evaluate


def get_neg_and_pos_lem(tweets):
    lem_pos = Tweet.get_all_messages_sentiment(tweets,Tweet.POSITIVE)
    lem_neg = Tweet.get_all_messages_sentiment(tweets,Tweet.NEGATIVE)
    lem_neu = Tweet.get_all_messages_sentiment(tweets,Tweet.NEUTRAL)

    all_pos = ' '.join(lem_pos)
    all_neg = ' '.join(lem_neg)
    all_neutral = ' '.join(lem_neu)
    return [[all_pos, all_neg, all_neutral], [lem_pos, lem_neg, lem_neu]]


def get_neg_and_pos_message(tweets):
    lem_pos = [tweet.message for tweet in tweets if tweet.sentiment == Tweet.POSITIVE]
    lem_neg = [tweet.message for tweet in tweets if tweet.sentiment == Tweet.NEGATIVE]
    lem_neu = [tweet.message for tweet in tweets if tweet.sentiment == Tweet.NEUTRAL]
    all_pos = ' '.join([x for x in lem_pos])
    all_neg = ' '.join([x for x in lem_neg])
    all_neutral = ' '.join([x for x in lem_neu])
    return [all_pos, all_neg, all_neutral]


def log_reg_train_data(tweets):
    train = np.array(tweets)
    x = 200
    while len(train) % x != 0:
        x += 1
    nfold = np.split(train, x)
    pred = np.array([])
    pred.shape = (0, 3)
    vectorizer = TfidfVectorizer()
    for i in range(0, len(nfold)):
        all_pos = ""
        all_neg = ""
        all_neutral = ""

        for j in range(0, len(nfold)):
            if i != j:
                [[pos, neg, neu], [n, p, nn]] = get_neg_and_pos_lem(nfold[j])
                all_pos += " " + pos
                all_neg += " " + neg
                all_neutral += " " + neu
        tfidf = vectorizer.fit_transform([all_pos, all_neg, all_neutral])
        m = vectorizer.transform(Tweet.get_all_messages(nfold[i]))
        pred = np.append(pred, tfidf.A.dot(m.T.A).T, axis=0)
    return pred


def get_tweets(data_file):
    tweets = []
    matrix = pd.read_csv(data_file, sep='\t').as_matrix()
    for tweet in matrix:
        tweets.append(Tweet(tweet[0], "", tweet[1], tweet[2]))
    return tweets


def normalise_fdist(fdist_list):
    normalised_list = []
    for x in fdist_list.most_common():
        y = (x[1] * 100) / len(fdist_list.most_common())
        _tuple = (x[0], y)
        normalised_list.append(_tuple)
    return normalised_list


def plot_fdist(pos_tok, neg_tok, neu_tok):
    all_tok = pos_tok + neu_tok + neg_tok
    fdist_all = FreqDist(all_tok)
    fdist_pos = FreqDist(pos_tok)
    fdist_neg = FreqDist(neg_tok)
    fdist_neu = FreqDist(neu_tok)
    show_bar_plot([i[0] for i in fdist_all.most_common(7)], [i[1] for i in fdist_all.most_common(7)],
                  "Frequency distribution of all tweets")
    show_bar_plot([i[0] for i in fdist_pos.most_common(7)], [i[1] for i in fdist_pos.most_common(7)],
                  "Frequency distribution of positive tweets")
    show_bar_plot([i[0] for i in fdist_neg.most_common(7)], [i[1] for i in fdist_neg.most_common(7)],
                  "Frequency distribution of negative tweets")
    show_bar_plot([i[0] for i in fdist_neu.most_common(7)], [i[1] for i in fdist_neu.most_common(7)],
                  "Frequency distribution of neutral tweets")


def plot_profanity_words(pos_mes, neg_mes, neu_mes):
    pro_filter = ProfanityFilter()
    cnt_pos = pro_filter.count_bad_words(pos_mes.lower())
    cnt_neg = pro_filter.count_bad_words(neg_mes.lower())
    cnt_neu = pro_filter.count_bad_words(neu_mes.lower())

    per_pos = sum(cnt_pos.values())
    per_neg = sum(cnt_neg.values())
    per_neu = sum(cnt_neu.values())

    cat = ('Positive', 'Negative', 'Neutral')
    performance = [per_pos, per_neg, per_neu]
    show_bar_plot(cat, performance, "Profanity frequency by sentiment")


plt.rcdefaults()

trainTweets = get_tweets('data/train-A.tsv')
testTweets = get_tweets('data/test-A-full.tsv')

vect = TfidfVectorizer()
[[posLem, negLem, neuLem], [pos_tokens, neg_tokens, neu_tokens]] = get_neg_and_pos_lem(trainTweets)
#[posMes, negMes, neuMes] = get_neg_and_pos_message(trainTweets)

#pos_list = [item for sublist in pos_tokens for item in sublist]
#neg_list = [item for sublist in neg_tokens for item in sublist]
#neu_list = [item for sublist in neu_tokens for item in sublist]
#print(pos_list)
#plot_fdist(pos_list, neg_list, neu_list)
#plot_profanity_words(posMes, negMes, neuMes)

tfidf = vect.fit_transform([posLem, negLem, neuLem])
m = vect.transform(Tweet.get_all_messages(trainTweets))
print("TF-IDF vectors (each column is a document):\n{}\nRows:\n{}".format(tfidf.T.A, vect.get_feature_names()))
# classifier = nltk.NaiveBayesClassifier.train(tfidf)
# print("Naive acc:", (nltk.classify.accuracy(classifier, m)) * 100)
#
# classifier.show_most_informative_features(4)

weights = vect.transform(Tweet.get_all_messages(testTweets))

res = [Tweet.sentiments[x] for x in np.argmax(tfidf.A.dot(weights.T.A), axis=0)]
print(np.argmax([1, 2, 3, 2, 11], axis=0))
print("Klasifikacijska tocnost: " + str(
    np.sum([1 if x == y else 0 for x, y in zip(res, Tweet.get_all_sentiment(testTweets))]) / len(res)))
print(Evaluate.evaluateA(res, Tweet.get_all_sentiment(testTweets)))

clf = RandomForestClassifier(n_estimators=25)
trainData = log_reg_train_data(trainTweets)
clf.fit(trainData, Tweet.get_all_sentiment(trainTweets))

res = clf.predict(tfidf.A.dot(weights.T.A).T)
np.sum([1 if x == y else 0 for x, y in zip(res, Tweet.get_all_sentiment(testTweets))]) / len(res)

print("Klasifikacijska tocnost regresija: " + str(
    np.sum([1 if x == y else 0 for x, y in zip(res, Tweet.get_all_sentiment(testTweets))]) / len(res)))

print(Evaluate.evaluateA(res, Tweet.get_all_sentiment(testTweets)))

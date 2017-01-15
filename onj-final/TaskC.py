from sklearn import linear_model

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from Tweet import Tweet
from sklearn.ensemble import RandomForestClassifier
import Evaluate



def get_negAndpos(tweets):
    lem_pos = [tweet.lemmatized for tweet in tweets if tweet.sentiment == Tweet.POSITIVE]
    lem_neg = [tweet.lemmatized for tweet in tweets if tweet.sentiment == Tweet.NEGATIVE]
    lem_neu = [tweet.lemmatized for tweet in tweets if tweet.sentiment == Tweet.NEUTRAL]
    lem_vneg = [tweet.lemmatized for tweet in tweets if tweet.sentiment == Tweet.VERYNEGATIVE]
    lem_vpos = [tweet.lemmatized for tweet in tweets if tweet.sentiment == Tweet.VERYPOSITIVE]

    all_pos = ' '.join([' '.join(x) for x in lem_pos])
    all_neg = ' '.join([' '.join(x)for x in lem_neg])
    all_neutral = ' '.join([' '.join(x) for x in lem_neu])
    all_vneg = ' '.join([' '.join(x) for x in lem_vneg])
    all_vpos = ' '.join([' '.join(x) for x in lem_vpos])
    return [all_pos, all_neg, all_neutral, all_vpos, all_vneg]

def logRegTrainData(tweets):
    train = np.array(tweets)
    nfold = np.split(train, 190)
    pred = np.array([])
    pred.shape = (0,5)
    for i in range(0, len(nfold)):
        all_pos = ""
        all_neg = ""
        all_neutral = ""
        all_vpos = ""
        all_vneg = ""
        for j in range(0, len(nfold)):
            if i != j:
                [pos, neg, neu, vpos, vneg] = get_negAndpos(nfold[j])
                all_pos += " " + pos
                all_neg += " " + neg
                all_neutral += " " + neu
                all_vpos += " " + vpos
                all_vneg += " " + vneg
        vect = TfidfVectorizer()
        tfidf = vect.fit_transform([all_pos, all_neg, all_neutral, vpos, vneg])
        m = vect.transform(Tweet.get_all_messages(nfold[i]))
        pred = np.append(pred, tfidf.A.dot(m.T.A).T, axis=0)
    return pred


def get_tweets(data_file):
    tweets = []
    matrix = pd.read_csv(data_file, sep='\t').as_matrix()
    for tweet in matrix:
        tweets.append(Tweet(tweet[0], tweet[1], tweet[2], tweet[3]))
    return tweets


trainTweets = get_tweets('data/train-CE.tsv')
testTweets = get_tweets('data/test-CE.tsv')


topics = np.unique(pd.read_csv('data/test-CE.tsv', sep='\t').as_matrix()[:,1])

vect = TfidfVectorizer()
tfidf = vect.fit_transform(get_negAndpos(trainTweets))

eval_res = []
eval_acc = []
for topic in topics:
    weights = vect.transform(Tweet.get_all_messages(testTweets, topic))

    res = [Tweet.sentiments[x] for x in np.argmax(tfidf.A.dot(weights.T.A), axis=0)]
    print("klasifikacijska tocnost " + str(
        np.sum([1 if x == y else 0 for x, y in zip(res, Tweet.get_all_sentiment(testTweets, topic))]) / len(res)))
    print(Evaluate.evaluateB(res, Tweet.get_all_sentiment(testTweets, topic)))
    eval_res += [Evaluate.evaluateB(res, Tweet.get_all_sentiment(testTweets, topic))]
    eval_acc += [np.sum([1 if x == y else 0 for x, y in zip(res, Tweet.get_all_sentiment(testTweets, topic))]) / len(res)]
print("Evaluation " + str(np.average(eval_res)))
print("Evaluation Acc" + str(np.average(eval_acc)))
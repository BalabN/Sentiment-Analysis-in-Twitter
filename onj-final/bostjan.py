import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn import linear_model


# Prepare text
def getTokens(text):
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    tokens = nltk.word_tokenize(text)
    return tokens


def get_negAndpos_test(matrix):
    all_pos = " ".join([x[2] for x in matrix if x[1] == 'positive'])
    all_neg = " ".join([x[2] for x in matrix if x[1] == 'negative'])
    all_neutral = " ".join([x[2] for x in matrix if x[1] == 'neutral'])

    lemmatizer = WordNetLemmatizer()
    all_pos = " ".join([lemmatizer.lemmatize(token) for token in getTokens(all_pos)])
    all_neg = " ".join([lemmatizer.lemmatize(token) for token in getTokens(all_neg)])
    all_neutral = " ".join([lemmatizer.lemmatize(token) for token in getTokens(all_neutral)])

    return [all_pos, all_neg, all_neutral]


def get_negAndpos(matrix):
    vect = TfidfVectorizer(ngram_range=(1, 5))

    nfold = np.split(matrix, 5)
    weights = np.array([])

    all_pos = ""
    all_neg = ""
    all_neutral = ""
    for j in range(0, len(nfold) - 1):
        all_pos += " " + " ".join([x[2] for x in nfold[j] if x[1] == 'positive'])
        all_neg += " " + " ".join([x[2] for x in nfold[j] if x[1] == 'negative'])
        all_neutral += " " + " ".join([x[2] for x in nfold[j] if x[1] == 'neutral'])

    lemmatizer = WordNetLemmatizer()
    all_pos = " ".join([lemmatizer.lemmatize(token) for token in getTokens(all_pos)])
    all_neg = " ".join([lemmatizer.lemmatize(token) for token in getTokens(all_neg)])
    all_neutral = " ".join([lemmatizer.lemmatize(token) for token in getTokens(all_neutral)])

    tfidf = vect.fit_transform([all_pos, all_neg, all_neutral])

    mat = [getTokens(x[2]) for x in nfold[4]]
    mat = [" ".join([lemmatizer.lemmatize(token) for token in x]) for x in mat]

    weights = vect.transform(mat).A

    return tfidf, vect, weights


x = pd.read_csv('data/train-A.tsv', sep='\t')
matrix_train = x.as_matrix()

tfidf, vect, m = get_negAndpos(matrix_train)
logistic = linear_model.LogisticRegression()
logistic.fit(m, np.split(matrix_train[:, 1], 5)[-1])

x = pd.read_csv('data/test-A.tsv', sep='\t')
matrix_test = x.as_matrix()

lemmatizer = WordNetLemmatizer()
mat = [getTokens(x[2]) for x in matrix_test]
mat = [" ".join([lemmatizer.lemmatize(token) for token in x]) for x in mat]

weights = vect.transform(mat).A

res = logistic.predict(weights)
print("klasifikacijska tocnost " + str(np.sum([1 if x == y else 0 for x, y in zip(res, matrix_test[:, 1])]) / len(res)))

classes = ['positive', 'negative', 'neutral']
res = [classes[x] for x in np.argmax(tfidf.A.dot(weights.T), axis=0)]
print("klasifikacijska tocnost brez log reg " + str(
    np.sum([1 if x == y else 0 for x, y in zip(res, matrix_test[:, 1])]) / len(res)))

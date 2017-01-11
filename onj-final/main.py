import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Prepare text
def getTokens():
   with open('shakespeare.txt', 'r') as shakes:
    text = shakes.read().lower()
    # remove punctuation
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    tokens = nltk.word_tokenize(text)
    return tokens


def get_negAndpos(matrix):
    all_pos = " ".join([x[2] for x in matrix if x[1] == 'positive'])
    all_neg = " ".join([x[2] for x in matrix if x[1] == 'negative'])
    all_neutral = " ".join([x[2] for x in matrix if x[1] == 'neutral'])

    return [all_pos, all_neg, all_neutral]

x = pd.read_csv('train-A.tsv', sep='\t')
matrix_train = x.as_matrix()



vect = TfidfVectorizer()
tfidf = vect.fit_transform(get_negAndpos(matrix_train))


x = pd.read_csv('test-A.tsv', sep='\t')
matrix_test = x.as_matrix()
weights = vect.transform(matrix_test[:,2])

classes = ['positive', 'negative', 'neutral']
res = [classes[x] for x in np.argmax(tfidf.A.dot(weights.T.A), axis=0)]
#print(classes[np.argmax(tfidf.A.dot(weights.T.A))])
print("klasifikacijska tocnost " + str(np.sum([1 if x==y else 0 for x,y in zip(res, matrix_test[:,1])])/len(res)))





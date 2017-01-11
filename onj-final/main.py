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

x = pd.read_csv('downloaded.tsv', sep='\t')
matrix = x.as_matrix()



vect = TfidfVectorizer()
tfidf = vect.fit_transform(get_negAndpos(matrix))

weights = vect.transform(["are yall really mad that he's paying $200 to fix a broken iPad, don't yall have bigger problems to deal with"])

classes = ['positive', 'negative', 'neutral']
print(classes[np.argmax(tfidf.A.dot(weights.T.A))])





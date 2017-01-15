from collections import Counter

import pandas as pd
import re


class ProfanityFilter(object):

    profanity = pd.read_csv('data/profanity_words.txt').as_matrix()
    profanity_string = ""

    def __init_(self):
        self.profanity = pd.read_csv('data/profanity_words.txt').as_matrix()
        self.profanity_string = ''.join(' '.join(x) for x in self.profanity)

    def count_bad_words(self, sentence):
        count = Counter()
        for w in self.profanity:
            for m in re.finditer(r"\b" + w[0] + r"\b", sentence):
                count[w[0]] += 1
        return count

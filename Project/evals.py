import nltk
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import numpy as np
import evaluate

def wups(words1, words2, alpha):

    sim = 1.0
    flag = False
    for w1 in words1:
        max_sim = 0
        for w2 in words2:
            word_sim = 0
            if w1 == w2:
                word_sim =  1.0
            else:
                w1_net = wordnet.synsets(w1)
                w1_len = len(w1_net)
                w2_net = wordnet.synsets(w2)
                w2_len = len(w2_net)
                if w2_len == 0 or w1_len==0: 
                    word_sim = 0.0
                else :
                    word_sim = w1_net[0].wup_similarity(w1_net[0])
                    if word_sim is None:
                        word_sim = 0.0

                    if word_sim < alpha:
                        word_sim = 0.1*word_sim
            if word_sim > max_sim:
                max_sim = word_sim
        if max_sim == 0: continue
        sim *= max_sim
        flag = True
    if not flag:
        sim = 0.0
    return sim


def get_wups(pred, truth, alpha):

    pred_tokens = word_tokenize(pred)
    truth_tokens = word_tokenize(truth)
    return min(wups(pred_tokens, truth_tokens, alpha),wups(truth_tokens, pred_tokens, alpha))
    

def levenshtein_distance(s1, s2):

    distances = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    for i in range(len(s1) + 1):
        distances[i][0] = i
    for j in range(len(s2) + 1):
        distances[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,  # Deletion
                distances[i][j - 1] + 1,  # Insertion
                distances[i - 1][j - 1] + cost,  # Substitution
            )

    return distances[-1][-1]

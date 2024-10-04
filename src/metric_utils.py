import numpy as np
from collections import Counter


def get_entropy(values):
    _, value_counts = np.unique(values, return_counts=True)
    value_probs = value_counts / len(values)
    return -np.sum(value_probs * np.log2(value_probs))


def get_avg(values):
    return sum(values) / len(values)


def get_majority_vote(values):
    return Counter(values).most_common(n=1)[0][0]


def get_majority_vote_rate(values):
    return Counter(values).most_common(n=1)[0][1] / len(values)

def get_majority_vote(values):
    return Counter(values).most_common(n=1)[0][0]
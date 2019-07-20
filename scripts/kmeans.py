# We used this script to check what is the best accuracy we can get from k-means
# Essentially we separate the data points into cluster and try to assign all permutations to class labels of clusters.

import functools
import itertools
import pickle
from collections import Counter

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

data = pickle.load(open('../logs/fb15k/sm.data.pkl', 'rb'))

kmeans = KMeans(n_clusters=6, n_init=20, max_iter=1000, n_jobs=10, verbose=0)
kmeans.fit(data[:, :-1])
ct = Counter(kmeans.labels_)

valid_data = pickle.load(open('../logs/fb15k/exp_words/sm_valid.pkl', 'rb'))
valid_pred = kmeans.predict(valid_data[:, :-1])


def calculate_accuracies(ycpu, ypred_cpu):
    labels = [x for x in range(5+1)]
    micro_f = metrics.f1_score(ycpu, ypred_cpu, labels=labels, average='micro')
    return micro_f


def permute(nclasses):
    permutations = list(itertools.permutations([i for i in range(nclasses)]))
    best_micro_f = 0.0
    for perm in permutations:
        valid_pred_loop = np.array([perm[i] for i in valid_pred])
        this_f = calculate_accuracies(valid_data[:, -1], valid_pred_loop)
        best_micro_f = max(best_micro_f, this_f)
        print(this_f, perm)
    print("Best Micro --> ", best_micro_f)


permute(6)

# Best --> 0.425287356322 (3, 0, 1, 2, 5, 4)

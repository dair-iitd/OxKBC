import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import pickle
from collections import Counter
import sys

data = pickle.load(open('../logs/fb15k-inv/exp_words/valid_annotated.pkl','rb'))

# kmeans = KMeans(n_clusters=6,n_init=20,max_iter=1000,n_jobs=10,verbose=0)
# kmeans.fit(data[:,:-1])
clf = SVC(gamma='auto')
clf.fit(data[:,:-1], data[:,-1])

ct = Counter(kmeans.labels_)

for key in ct:
    ct[key]  = ct[key]*1.0/len(data)
print(ct)
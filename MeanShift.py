import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use("ggplot")


X = np.array([[1, 2],
              [3, 5],
              [5, 2 ],
              [1, 1],
              [5,2],
              [6,3],
              [9,9],
              [11,6],
              [3,3],
              [10,9]])


colors = 10*["g","r","c","b","k"]

class Mean_Shift:
    def __init__(self, radius=4):
        self.radius = radius

    def fit(self, spongebob_data):
        centroids = {}

        for i in range(len(spongebob_data)):
            centroids[i] = spongebob_data[i]
        
        while True:
            new_centroids = []
            new_centroids_two = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in spongebob_data:
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids_two.append(new_centroid)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))
            
            prev_centroids = (centroids)

            centroids_test = {}
            for i in range(len(new_centroids_two)):
            	centroids_test[i] = new_centroids_two[i]

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True
            print (centroids_test)
            print (prev_centroids)

            for i in centroids_test:
                if not np.array_equal(centroids_test[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
                
            if optimized:
                break

        self.centroids = centroids



clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

plt.scatter(X[:,0], X[:,1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='b', marker='o')

plt.show()		

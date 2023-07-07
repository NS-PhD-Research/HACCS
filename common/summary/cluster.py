"""
cluster.py

Clustering algorithms for flsys
UMN DCSG, 2021
"""

import numpy as np
from numpy import unique
from numpy.linalg import norm

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

def hdist(v1, v2):

    sqrt2 = np.sqrt(2)

    sqrtdevP = np.sqrt(v1)
    sqrtdevQ = np.sqrt(v2)
    sumOfSqrOfDiffSqrtdevice = np.sum((sqrtdevP - sqrtdevQ) ** 2)
    return np.sqrt(sumOfSqrOfDiffSqrtdevice) / sqrt2

def mnorm(m1, m2):

    avgDist = 0.0
    for x in range(m1.shape[0]):
        d = hdist(m1[x,:], m2[x,:])
        avgDist += d
    return avgDist
    #return avgDist / float(m1.shape[0])

"""
Basic clustering routine for clustering histograms.
Searches for clusters of at least 2 points. Returns an array of assignments
with a -1 indicating that no suitable cluster was found for that device.
"""
def cluster_hist(histogramList, keySpace):

    X = np.zeros((len(histogramList), len(keySpace)))
    for idx, hist in enumerate(histogramList):
        X[idx,:] = hist.toFrequencyArray(keySpace)

    dim = len(histogramList)
    distMat = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i, dim):
            #dist = hdist(X[i,:], X[j,:])
            dist = norm(X[i,:] - X[j,:])
            distMat[i,j] = dist
            distMat[j,i] = dist

    return OPTICS(min_samples=2,cluster_method="dbscan",eps=0.25,
                  metric="precomputed").fit_predict(distMat)
    #return OPTICS(min_samples=2,
    #              metric="precomputed").fit_predict(distMat)

"""
Basic clustering routine for clustering a list of histograms.
Searches for clusters of at least 2 points. Returns an array of assignments
with a -1 indicating that no suitable cluster was found for that device.
"""
def cluster_mat(matList, xKeySpace, yKeySpace):

    dim = len(matList)
    distMat = np.zeros((dim, dim))

    mats = []
    for i in range(dim):
        mat = matList[i].toMatrix(xKeySpace, yKeySpace)
        mats.append(mat)

    for i in range(dim):
        for j in range(i, dim):
            dist = mnorm(mats[i], mats[j])
            distMat[i,j] = dist
            distMat[j,i] = dist

    return OPTICS(min_samples=2,
                  metric="precomputed").fit_predict(distMat)

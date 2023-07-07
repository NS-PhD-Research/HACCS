"""
hist.py

Histogram summary implementation for flsys
UMN DCSG, 2021
"""

import numpy as np
import json

from collections import Counter

class HistSummary(object):

    """
    @param labels   List of all labels in the local training data.
                    A histogram will be constructed by counting the
                    instances of each label in the list.
    """
    def __init__(self, labels=[]):

        if not isinstance(labels, list):
            raise ValueError("Labels must be a list")

        self.values = Counter(labels)

    """
    @brief Adds noise to the histogram satisfying (epsilon, 0) - DP
    """
    def addNoise(self, epsilon):
        
        for k in self.values.keys():

            noise = float(np.random.laplace(0, 1/epsilon, 1))
            newValue = self.values[k] + noise

            if newValue < 0.0:
                # Change to indicator variable so Jaccard Similarity can use it
                self.values[k] = min(self.values[k], 1.0)
            else:
                self.values[k] = newValue

    def at(self, key):
        return self.values[key]

    def getKeys(self):
        return set(self.values.keys())

    def computeJaccardSimilarity(self, hist2):

        if not isinstance(hist2, HistSummary):
            raise ValueError("Input must be a HistSummary")

        keys1 = set(self.values.keys())
        keys2 = set(hist2.values.keys())

        return float(len(keys1.intersection(keys2))) / float(len(keys1.union(keys2)))

    def computeHellingerDist(self, hist2, allKeys=[]):

        if not isinstance(hist2, HistSummary):
            raise ValueError("Input must be a HistSummary")

        sqrt2 = np.sqrt(2)
        if len(allKeys) < 1:
            allKeys = (Counter(self.values) + Counter(hist2.values)).keys()

        devP = self.toFrequencyArray(allKeys)
        devQ = hist2.toFrequencyArray(allKeys)

        sqrtdevP = np.sqrt(devP)
        sqrtdevQ = np.sqrt(devQ)
        sumOfSqrOfDiffSqrtdevice = np.sum((sqrtdevP - sqrtdevQ) ** 2)
        return np.sqrt(sumOfSqrOfDiffSqrtdevice) / sqrt2

    """
    Serialization functions
    """
    def toJson(self):
        return json.dumps(self.values)

    def fromJson(self, jsonStr):
        self.values = json.loads(jsonStr)

    def __str__(self):
        return self.toJson()

    def toArray(self, keySpace):

        arr = np.zeros(len(keySpace))
        arrIdx = 0

        for key in keySpace:
            if key in self.values.keys():
                arr[arrIdx] = float(self.values[key])
            arrIdx += 1

        return arr

    def toFrequencyArray(self, keySpace):

        arr = self.toArray(keySpace)
        denom = 0.0
        for key in self.values.keys():
            denom += float(self.values[key])

        return arr / denom


"""
Histogram maxtrix summary implementation for flsys
"""
class HistMatSummary(object):

    """
    @param hists    A dict where the keys correspond to response lables
                    and the value is a histogram with the data distribution
                    for that label.
    """
    def __init__(self, hists={}):

        if not isinstance(hists, dict):
            raise ValueError("hists must be a dict")

        self.values = hists

    """
    @brief Adds noise to each histogram satisfying (epsilon, 0) - DP
    """
    def addNoise(self, epsilon):
        
        for k in self.values.keys():
            self.values[k].addNoise(epsilon)

    def at(self, key):
        return self.values[key]

    def getKeys(self):
        return set(self.values.keys())

    def getYKeys(self):
        return self.getKeys()

    def getXKeys(self):
        xkeys = set()
        for k in self.values.keys():
            hk = self.values[k]
            xkeys = xkeys.union(hk.getKeys())
        return xkeys
    """
    Serialization functions
    """
    def toJson(self):

        tmpDict = {}
        for k in self.values.keys():
            tmpDict[k] = self.values[k].toJson()

        return json.dumps(tmpDict)

    def fromJson(self, jsonStr):

        tmpDict = json.loads(jsonStr)
        self.values = {}

        for k in tmpDict.keys():
            hs = HistSummary()
            hs.fromJson(tmpDict[k])
            self.values[k] = hs

    def __str__(self):
        return self.toJson()

    def toMatrix(self, xKeySpace, yKeySpace):

        arr = np.zeros((len(yKeySpace), len(xKeySpace)))
        rowIdx = 0
        colIdx = 0

        myYKeys = self.values.keys()

        for yKey in yKeySpace:
            if yKey in myYKeys:

                colIdx = 0
                rowSum = 0.0
                myXKeys = self.values[yKey].getKeys()
                for xKey in xKeySpace:
                    if xKey in myXKeys:
                        val = float(self.values[yKey].at(xKey))
                        arr[rowIdx, colIdx] = val
                        rowSum += val

                    colIdx += 1
                arr[rowIdx,:] /= rowSum
            rowIdx += 1

        return arr


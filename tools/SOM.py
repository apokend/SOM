import os
import numpy as np
import bisect
import somoclu
import functools
import timeit
import datetime
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
try:
    from functions import runTime, scoreExport, create_path
except ImportError:
    from .functions import runTime, scoreExport, create_path


class SOM(BaseEstimator, ClassifierMixin):

    def __init__(self, dim=50, epochs=15, logPath=None):
        if logPath == None:
            logPath = create_path('logs')
        if not os.path.exists(logPath):
            os.makedirs(logPath)
        self.log_path = logPath
        self.dim = dim
        self.epochs = epochs

    def __calc_treshold(self, distance, lvl_trust=0.99):
        hist, bins = np.histogram(distance, bins='auto', density=True)
        pdf = [0]
        pdf.extend(hist)

        prob = [0]
        for elem in pdf[1:]:
            value = prob[-1] + elem * bins[1]
            prob.append(value)

        value = list(filter(lambda x: x > lvl_trust, prob))[0]
        indx = prob.index(value)
        treshold = bins[indx]
        return treshold

    def __get_idx(self, L, n):
        i = bisect.bisect_left(L, n)
        if i:
            return i - 1
        else:
            return 0

    #@runTime("log_path")
    def fit(self, X, y=None):
        print('Fit method enable')
        self.som = somoclu.Somoclu(
            self.dim, self.dim, gridtype='hexagonal', initialization="pca")
        self.som.train(data=X, epochs=self.epochs)
        self.distances = [min(x) for x in self.som.get_surface_state(X)]
        self.t = self.__calc_treshold(self.distances)
        return self

    #@runTime("log_path")
    def predict(self, X, y=None):
        print('Predict method enable')
        self.distances_test = [min(x) for x in self.som.get_surface_state(X)]
        preds, probs = [], []
        sorted_train = sorted(self.distances)
        for d in self.distances_test:
            idx = self.__get_idx(sorted_train, d)
            probs.append(idx / len(self.distances))
            if d > self.t:
                preds.append(1)
            else:
                preds.append(0)
        self.preds, self.probs = preds, probs
        return preds, probs

    #@scoreExport("log_path")
    def score(self, y_test, preds = None, probs = None):
        if preds == None:
            preds = self.preds
        if probs== None:
            probs=self.probs
        scores = {}
        for score in [accuracy_score, precision_score, recall_score, f1_score]:
            metric = score.__name__.split('_')[0]   
            quality = score(y_test, preds)
            scores[metric] = quality  
            print(metric + ':%.3f'%quality)
        scores['roc-auc'] = roc_auc_score(y_test, probs)
        print('roc-auc: %.3f'%scores['roc-auc'])
        return scores

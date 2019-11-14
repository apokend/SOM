#---------------------------+
#        Version:  1.01     +
#   Status: Ready to Test   +
#   Author: Shevchenko A.A. +
#-------------------------- +

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
    from functions import runTimeLogger
    from logger import logger
except ImportError:
    from .functions import runTimeLogger
    from .logger import logger


class SOM(BaseEstimator, ClassifierMixin):

    def __init__(self, dim=50, epochs=15):
        logger.info('<< SOM Model: INIT >>')
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

    @runTimeLogger
    def fit(self, X, y=None):
        logger.info('<< SOM Model | Fit Method: RUN >>')
        self.som = somoclu.Somoclu(
            self.dim, self.dim, gridtype='hexagonal', initialization="pca")
        self.som.train(data=X, epochs=self.epochs)
        self.distances = [min(x) for x in self.som.get_surface_state(X)]
        self.t = self.__calc_treshold(self.distances)
        logger.info('<< SOM Model | Fit Method: DONE >>')
        return self

    @runTimeLogger
    def predict(self, X, y=None):
        logger.info('<< SOM Model | Predict Method: RUN >>')
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
        logger.info('<< SOM Model | Predict Method: DONE >>')
        return preds, probs
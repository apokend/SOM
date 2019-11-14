#---------------------------+
#        Version:  1.01     +
#   Status: Ready to Test   +
#   Author: Shevchenko A.A. +
#-------------------------- +

import sys
import os
import timeit
from datetime import datetime as dt
from datetime import timedelta
import functools
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
try:
    from logger import logger
except ImportError:
    from .logger import logger


def runTimeLogger(function):
    def wrapper(self, *args, **kwargs):
        start = timeit.default_timer()
        r = function(self, *args, **kwargs)
        end = timeit.default_timer()
        timer = timedelta(seconds=end - start)
        name = function.__name__
        logger.info(f'<< Runtime {name} method: {timer} >>')
        return r
    return wrapper


def collect_data(path=None):
    if path == None:
        path = os.getcwd().replace('\\','/') + '/dataset/data'
    DFs = []
    for file in os.listdir(path):
        try:
            data = pd.read_csv(path+'/'+file, '\t', skiprows=1)
        except pd.errors.ParserError:
            continue
        except UnicodeDecodeError:
            data = pd.read_excel(path+'/'+file, skiprows=1)  
        if data.shape[1] == 24:
            data.drop(columns=['wIO_sec(disk)', 'rIO_sec(disk)'], inplace=True)
        if 'Unnamed' in data.columns[-1]:
            data.drop(columns=[data.columns[-1]], inplace=True)
        if 'anomaly' in file:
            data['label'] = 1
        elif 'normal' in file:
            data['label'] = 0

        try:
            DFs.append(data.drop(columns=['num', 'Time']))
        except:
            DFs.append(data.drop(columns=['num', 'ts']))
        
    data = pd.concat(DFs, ignore_index=True)
    data.dropna(inplace=True)
    data.to_csv('dataset/dataset.csv', index=False)

def test_train_split(X, y, test_size = None):
    principalDf = pd.DataFrame(data = X, columns = ['pc1', 'pc2', 'pc3', 'pc4'])
    data = pd.concat([principalDf, y.label], axis = 1)
    good = data[data.label == 0]
    good = shuffle(good)
    CONST_PROC = 20000 if test_size == None else int(good.shape[0] *(1 - test_size))
    good_train = good.iloc[:CONST_PROC,:-1].values
    good_test = good.iloc[CONST_PROC:,:-1].values
    bad = data[data.label == 1].values[:,:-1]

    X_train = good_train
    X_test = np.concatenate((good_test, bad), axis=0)
    y_train = np.array([0]*good_train.shape[0])
    y_test = np.array([0]*good_test.shape[0] + [1]*bad.shape[0])
    return X_train, X_test, y_train, y_test


def score(y, preds, probs):
    logger.info('<< Getting scores of metrics: ACTIVATE | Status: IN PROCCESS >>')
    scores = {}
    for score in [accuracy_score, precision_score, recall_score, f1_score]:
        name = score.__name__.split('_')[0]
        scores[name] = score(y, preds)
    scores['roc-auc'] = roc_auc_score(y, probs)
    logger.info(
        "<< Getting scores of metrics: ACTIVATE | Status: DONE >>")
    logger.info('\n'+
        "\n---------------\n".join([f"{k} : {v}" for k, v in scores.items()]))

def export_labels(labels,file_name):
    print(labels, file = open(file_name,'w'))
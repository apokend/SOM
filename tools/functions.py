import sys
import os
import timeit
from datetime import datetime as dt
from datetime import timedelta
import functools
import numpy as np
import pandas as pd
from sklearn.utils import shuffle



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

def test_train_split(X, y):
    principalDf = pd.DataFrame(data = X, columns = ['pc1', 'pc2', 'pc3', 'pc4'])
    data = pd.concat([principalDf, y.label], axis = 1)
    good = data[data.label == 0]
    good = shuffle(good)
    good_train = good.iloc[:20000,:-1].values
    good_test = good.iloc[20000:,:-1].values
    bad = data[data.label == 1].values[:,:-1]

    X_train = good_train
    X_test = np.concatenate((good_test, bad), axis=0)
    y_train = np.array([0]*good_train.shape[0])
    y_test = np.array([0]*good_test.shape[0] + [1]*bad.shape[0])
    return X_train, X_test, y_train, y_test


def runTime(attribute):
    def _runTime(method_to_decor):
        @functools.wraps(method_to_decor)
        def wrapper(self, *args, **kwargs):
            path = getattr(self, attribute)
            f = list(sys._current_frames().values())[0]
            f = f.f_back.f_globals['__file__'].split('\\')[-1]
            file_name = str(f.split('.')[0]).upper()
            date = dt.now().strftime("%d-%b-%Y")
            folder_path = f"{path}/{date}/{file_name}/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            time_event = dt.now().strftime("%H-%M-%S")
            print(f"{dt.now()}: <<Called '{method_to_decor.__name__}' method>>",
                  file=open(folder_path + f"/{time_event}_{file_name}_{method_to_decor.__name__}.txt", 'a'))
            start = timeit.default_timer()
            r = method_to_decor(self, *args, **kwargs)
            end = timeit.default_timer()
            timer = timedelta(seconds=end - start)
            print(f"{dt.now()}: <<Method '{method_to_decor.__name__}' -- runtime: {timer} >>",
                  file=open(folder_path + f"/{time_event}_{file_name}_{method_to_decor.__name__}.txt", 'a'))
            return r
        return wrapper
    return _runTime



def scoreExport(attribute):
    def _scoreExport(method_to_decor):
        @functools.wraps(method_to_decor)
        def wrapper(self, *args, **kwargs):
            path = getattr(self, attribute)
            f = list(sys._current_frames().values())[0]
            f = f.f_back.f_globals['__file__'].split('\\')[-1]
            file_name = str(f.split('.')[0]).upper()
            date = dt.now().strftime("%d-%b-%Y")
            folder_path = f"{path}/{date}/{file_name}/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            time_event = dt.now().strftime("%H-%M-%S")
            r = method_to_decor(self, *args, **kwargs)
            file_path = folder_path + f"/{time_event}_{file_name}_{method_to_decor.__name__}.txt"
            print(f"{dt.now()}: <<Called '{method_to_decor.__name__}' method >>",
                  file=open(file_path, 'a'))
            print("-"*19,file=open(file_path, 'a'))
            print("Metric \t\tQuality",file=open(file_path, 'a'))
            print("-"*19,file=open(file_path, 'a'))
            for k,v in r.items():
                print(f"* {k}:\t{v:.3f}\n",
                  file=open(file_path,'a'))
            return r
        return wrapper
    return _scoreExport



def create_path(x:str) -> str:
    if not x.startswith('/'):
        x = '/'+x
    return os.getcwd().replace('\\','/') + x
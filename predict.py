from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.externals import joblib
import pandas as pd
import os

print('PREDICT method: ACTIVE')
print('**** Getting path ****')

path = os.getcwd().replace('\\','/')
print('Path: READY')

model = joblib.load(path+"/model/som_predictor.pkl")
print('Model: READY')

data = pd.read_csv('dataset/dataset.csv')
print('Data: READY')

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

print('Predict: Start')
preds, probs = model.predict(X)

print('Getting scores...')
for score in [accuracy_score, precision_score, recall_score, f1_score]:
        print(score.__name__.split('_')[0]+':%.3f'%(score(y, preds)))
print('roc-auc: %.3f'%(roc_auc_score(y, probs)))

print('END')
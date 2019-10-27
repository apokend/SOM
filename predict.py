from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.externals import joblib
import pandas as pd
import os

path = os.getcwd().replace('\\','/')
model = joblib.load(path+"/model/som_predictor.pkl")
data = pd.read_csv('dataset/dataset.csv')

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

preds, probs = model.predict(X)

for score in [accuracy_score, precision_score, recall_score, f1_score]:
        print(score.__name__.split('_')[0]+':%.3f'%(score(y, preds)))
print('roc-auc: %.3f'%(roc_auc_score(y, probs)))

import os 
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from tools.pipelines import *
from tools.functions import *
from tools.SOM import SOM

"""
print('---- Success ----')
print('Trying to find dataset....')

if not 'dataset' in os.listdir('dataset'):
	print('---- Dataset not found---')
	print(' //-- Creating dataset.. --\\\\')
	collect_data()
	print('<< Dataset created >> ')
else:
	print('---- Dataset is found ----')
"""

data = pd.read_csv('dataset/dataset.csv')
X = data.iloc[:,:-1].values
X = prep_pipeline.fit_transform(X)
X_train, X_test, y_train, y_test = test_train_split(X, data)

print(""" Data is ready
Model created
Starting learning...""")

clf = SOM()
clf.fit(X_train)


print('Model prediction')
preds, probs = clf.predict(X_test)
print('Success')

print('Geting scores')
clf.score(y_test)

"""
for score in [accuracy_score, precision_score, recall_score, f1_score]:
		 print(score.__name__.split('_')[0]+':%.3f'%(score(y_test, preds)))
print('roc-auc: %.3f'%(roc_auc_score(y_test, probs)))
"""
print('Saving model')
full_pipeline_with_predictor.steps.append(['SOM',clf])
joblib.dump(full_pipeline_with_predictor, "model/som_predictor.pkl")
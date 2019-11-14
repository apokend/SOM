#---------------------------+
#        Version:  1.01     +
#   Status: Ready to Test   +
#   Author: Shevchenko A.A. +
#-------------------------- +

import os 
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from tools.pipelines import *
from tools.functions import *
from tools.SOM import SOM

logger.info('<< Train Logger: INIT >>')

logger.info('<< Train mode: ACTIVATE | Status: Trying to find dataset >>')
if not 'dataset.csv' in os.listdir('dataset'):
	logger.info('<< Train mode: ACTIVATE | Status: Dataset not found | Creating dataset >>')
	collect_data()
	logger.info('<< Train mode: ACTIVATE | Status: Dataset created >>')
else:
	logger.info('<< Train mode: ACTIVATE | Status: Dataset is found | Loading data >>')

data = pd.read_csv('dataset/dataset.csv')
X = data.iloc[:,:-1].values
X = prep_pipeline.fit_transform(X)
X_train, X_test, y_train, y_test = test_train_split(X, data)

clf = SOM()
clf.fit(X_train)
preds, probs = clf.predict(X_test)
score(y_test, preds = preds, probs = probs)

logger.info('<< Train mode: ACTIVATE | Status: Saving model >>')
full_pipeline_with_predictor.steps.append(['SOM',clf])
joblib.dump(full_pipeline_with_predictor, "model/som_predictor.pkl")
logger.info('<< Train mode: ACTIVATE | Status: Model saved >>')
logger.info('<< Train Logger: END >>')
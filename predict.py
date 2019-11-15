#---------------------------+
#        Version:  1.02     +
#   Status: Ready to Prod   +
#   Author: Shevchenko A.A. +
#-------------------------- +

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.externals import joblib
import pandas as pd
import os
from tools.logger import logger
from tools.functions import score, export_labels

def main():
    # Get path
    path = os.getcwd().replace('\\', '/')
    logger.info("<< Model Preparation: IN PROCESS | Status: [Path : READY, ] >>")
    # Load model
    model = joblib.load(path + "/model/som_predictor.pkl")
    logger.info("<< Model Preparation: IN PROCESS | Status: [Path : READY, Model : READY, ] >>")
    # Load data
    data = pd.read_csv('dataset/dataset.csv')
    logger.info("<< Model Preparation: IN PROCESS | Status: [Path : READY, Model : READY, Data : READY] >>")
    
    # Getting data 
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # All that we need for pre-processing new data is allready in the model Pipeline
    logger.info("<< Model Prepatation: DONE | PREDICT method: ACTIVE >>")
    preds, probs = model.predict(X)
    logger.info(
        "<< Model Prepatation: DONE | PREDICT method: DONE >> ")

    # Here we can get and log our scores based on a set of standart metrics 
    score(y, preds = preds, probs = probs)
    
    # Need to add label export
    logger.info("<< EXPORT LABELS >>")
    export_labels(preds,'labels.txt')

if __name__ == '__main__':
    logger.info('<< Logger: INIT >>')
    main()
    logger.info('<< Logger: END >>')

#---------------------------+
#        Version:  1.01     +
#   Status: Ready to Test   +
#   Author: Shevchenko A.A. +
#-------------------------- +

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.externals import joblib
import pandas as pd
import os
from tools.logger import logger
from tools.functions import score, export_labels

def main():
    path = os.getcwd().replace('\\', '/')
    logger.info("<< Model Preparation: IN PROCESS | Status: [Path : READY, ] >>")

    model = joblib.load(path + "/model/som_predictor.pkl")
    logger.info("<< Model Preparation: IN PROCESS | Status: [Path : READY, Model : READY, ] >>")

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
    """
    scores = {}
    for score in [accuracy_score, precision_score, recall_score, f1_score]:
        name = score.__name__.split('_')[0]
        scores[name] = score(y, preds)
    scores['roc-auc'] = roc_auc_score(y, probs)

    logger.info(
        "PREDICT method: ACTIVE | Status: READY | Prediction: READY | Scores: READY")
    logger.info(
        "\n---------------\n".join([f"{k} : {v}" for k, v in scores.items()]))
    """
### Need to add label export
    logger.info("<< EXPORT LABELS >>")
    export_labels(preds,'labels.txt') #TODO FUN

if __name__ == '__main__':
    logger.info('<< Logger: INIT >>')
    main()
    logger.info('<< Logger: END >>')

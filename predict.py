from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.externals import joblib
import pandas as pd
import os
from tools.logger import logger


def main():
    print('PREDICT method: ACTIVE')
    print('**** Getting path ****')
    path = os.getcwd().replace('\\', '/')
    print('Path: READY')
    logger.info("PREDICT method: ACTIVE | Status: [Path : READY, ] ")

    model = joblib.load(path + "/model/som_predictor.pkl")
    print('Model: READY')
    logger.info("PREDICT method: ACTIVE | Status: [Model : READY, ] ")

    data = pd.read_csv('dataset/dataset.csv')
    print('Data: READY')
    logger.info("PREDICT method: ACTIVE | Status: [Data : READY, ] ")

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    logger.info("PREDICT method: ACTIVE | Status: READY | Prediction... ")
    print('Predict: Start')
    preds, probs = model.predict(X)

    logger.info(
        "PREDICT method: ACTIVE | Status: READY | Prediction: READY | Scores ... ")

    print('Getting scores...')
    """
    scores = {}
    for score in [accuracy_score, precision_score, recall_score, f1_score]:
    	name = scores[score.__name__.split('_')[0]]
        scores[name] = score(y, preds)
        print(name + ':%.3f' % (scores[name]))
    scores['roc-auc'] = roc_auc_score(y, probs)
    print('roc-auc: %.3f' % (scores['roc-auc']))
    """
    scores = model.score(y, preds, probs)
    logger.info(
        "PREDICT method: ACTIVE | Status: READY | Prediction: READY | Scores: READY")
    logger.info("\n---------------\n".join([f"{k} : {v}" for k, v in data.items()]))


if __name__ == '__main__':
    logger.info('Logger: INIT')
    main()

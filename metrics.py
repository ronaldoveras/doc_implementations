from scipy.stats import mstats

import data
import pandas as pd
import numpy as np

def get_median(results):
    results.sort()
    return (results[int(len(results) / 2)] + results[int((len(results) - 1) / 2)]) / 2


def predict(cls,df_features):
    prediction = predict_proba(cls, df_features=df_features)
    prediction['prediction'] = (
            prediction['probability'] >= .5).round().astype('int')
    return prediction


# def predict_proba(cls, df_features):
#     prediction = cls.base_learner.predict_proba(df_features)
#     return prediction


def predict_proba(cls, df_features):

        X = df_features[data.FEATURES].values

        probabilities = cls.base_learner.predict_proba(X)
        probabilities = probabilities[:, 1]


        probability = df_features.copy()
        probability['probability'] = probabilities
        return probability
def prequential_recalls(results, fading_factor):
    recalls = []
    counts = np.zeros(2)
    hits = np.zeros(2)
    targets = results['target']
    predictions = results['prediction']
    n_samples = len(targets)
    for i in range(n_samples):
        label = targets[i]
        counts[label] = 1 + fading_factor * counts[label]
        hits[label] = int(label == predictions[i]) + \
            fading_factor * hits[label]
        recalls.append(hits / (counts + 1e-12))
    columns = ['r{}'.format(i) for i in range(2)]
    recalls = pd.DataFrame(recalls, columns=columns)
    return pd.concat([results, recalls], axis='columns')


def prequential_recalls_difference(recalls):
    recalls_difference = recalls.copy()
    recalls_difference['r0-r1'] = (recalls['r0'] - recalls['r1']).abs()
    return recalls_difference

def prequential_gmean(recalls):
    gmean = mstats.gmean(recalls[['r0', 'r1']], axis=1)
    gmean = pd.DataFrame(gmean, columns=['g-mean'])
    return pd.concat([recalls, gmean], axis='columns')
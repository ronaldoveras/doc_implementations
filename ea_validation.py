import random
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
from scipy.stats import mstats
from sklearn.metrics import recall_score
import shap

warnings.filterwarnings("ignore")


def random_state_seed(seed):
    return seed & (2**32 - 1)

np.random.seed(random_state_seed(615958507449322123931599253290210168))

def recall_script(results):
    count_defects = results[results['target'] == 1].shape[0]
    count_non_defects = results[results['target'] == 0].shape[0]
    count_inspect_defects = 0
    count_inspect_non_defects = 0
    for _, row in results.iterrows():
        if row['prediction'] == 1:
            if row['target'] == 1:
                count_inspect_defects = count_inspect_defects + 1
        if row['prediction'] == 0:
            if row['target'] == 0:
                count_inspect_non_defects = count_inspect_non_defects + 1
    recall = count_inspect_defects / count_defects
    print('Recall script - Classe 1 {} %'.format(recall * 100))
    recall_0 = count_inspect_non_defects / count_non_defects
    print('Recall script - Classe 0 {} %'.format(recall_0 * 100))

    recall_geral = (count_inspect_non_defects + count_inspect_defects) / (count_non_defects + count_defects)
    print('Recall script - Classe geral {} %'.format(recall_geral * 100))

def recall(y_true, y_pred):


  tp = np.sum(y_true == y_pred)
  fn = np.sum(y_true != y_pred)
  return tp / (tp + fn)

def prequential_gmean(recalls):
    gmean = mstats.gmean(recalls[['r0', 'r1']], axis=1)
    gmean = pd.DataFrame(gmean, columns=['g-mean'])
    return pd.concat([recalls, gmean], axis='columns')

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

def ea_evaluate(effort_limit, target_prediction_test):
        # Calcula a quantidade de linhas de código total (esforço)
        sum_loc = target_prediction_test['ld'].sum() + target_prediction_test['la'].sum()
        # print('Total de LOC {}'.format(sum_loc))

        # Calcula o limite de linhas de código a serem inspecionadas
        threshold = sum_loc * effort_limit
        # print('Threshold {}'.format(threshold))

        # Conta a quantidade de mudanças que são defeitos e as que não são
        count_defects = target_prediction_test[target_prediction_test['target'] == 1].shape[0]
        count_non_defects = target_prediction_test[target_prediction_test['target'] == 0].shape[0]

        # Definição de variáveis de controle
        cum_effort = 0
        count_inspect = 0
        count_inspect_defects = 0
        idx_k = 0  # variável igual ao IFA
        ja_encontrou_k = False
        # Cria coluna 'loc' referente ao esforço
        # target_prediction_test['loc'] = target_prediction_test['la'] + target_prediction_test['ld']
        # Ordena o dataset por predição e depois pelo esforço
        # target_prediction_test = target_prediction_test.sort_values(['prediction', 'loc'], ascending=[False, False])


        for _, row in target_prediction_test.iterrows():
            cum_effort = cum_effort + row['loc']  # acumula o esforço
            count_inspect = count_inspect + 1  # incrementa a quantidade de inspecionados
            #    print('Exemplo atual {},{}'.format(row['target'], row['prediction']))
            if row['prediction'] == 1:
                count_inspect_defects, idx_k, ja_encontrou_k = calcular_k(count_inspect_defects, idx_k, ja_encontrou_k, row)
            if cum_effort >= threshold:  # momento de finalizar a inspeção
                #    print('Esforço cumulativo {}'.format(cum_effort))
                #    print('Quantidade inspecionados {}'.format(count_inspect))
                #    print('Quantidade de defeitos inspecionados {}'.format(count_inspect_defects))
                precision = count_inspect_defects / count_inspect

                print('Precision {} %'.format(precision * 100))
                recall = count_inspect_defects / count_defects
                print('Recall {} %'.format(recall * 100))
                f1 = (2 * precision * recall) / (precision + recall)
                print('F1 {} %'.format(f1 * 100))
                print('ifa {}'.format(idx_k))
                print('PCI@20% {}'.format(count_inspect / target_prediction_test.shape[0]))

                break


def calcular_k(count_inspect_defects, idx_k, ja_encontrou_k, row):
    if row['target'] == 0 and ja_encontrou_k == False:  # enquanto falsos positivos forem encontrados, conta-se
        idx_k = idx_k + 1
    elif row['target'] == 1:
        count_inspect_defects = count_inspect_defects + 1
        if ja_encontrou_k == False:
            ja_encontrou_k = True
    return count_inspect_defects, idx_k, ja_encontrou_k


path = '/home/s007840944/#Pessoal/#Doutorado/#DinaldoAP/dissertation/jit-sdp-nn/test-verify.csv'
df = pd.read_csv(path, header=0)
# rcls = prequential_recalls(df, .99)
# print(rcls.head())
print('Recall off {}%'.format(recall(df['target'].values, df['prediction'].values)*100))
recall_script(df)
print('recall sklearn {}'.format(recall_score(df['target'].values, df['prediction'].values, average=None)))

ea_evaluate(.99, df)

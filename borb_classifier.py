import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

import data
import metrics
import sampler_data


def create_nb_model():
    classifier = GaussianNB()
    return classifier

def create_lr_model():
    lr_alpha=0.07834780974820314
    l1_ratio = 0.15
    classifier = SGDClassifier(loss='log', penalty='elasticnet',
                               alpha=lr_alpha, l1_ratio=l1_ratio, shuffle=False)
    return classifier

class BORB():
    def __init__(self, max_sample_size, th, l0, l1, m, n_updates):
        self.base_learner = create_lr_model()
        self.max_sample_size = max_sample_size
        self.th = th
        self.l0 = l0
        self.l1 = l1
        self.m = m
        self.n_updates = n_updates

    def train(self, df_train, **kwargs):
        # print('XXXXXXX REALIZANDO TREINAMENTO BORB')
        lambda0 = 1
        lambda1 = 1
        df_ma = kwargs.pop('df_ma', None)
        self.ma = self.th
        # print('---> self.classifier_train.n_iterations {}'.format(self.n_iterations))
        df_train[data.TARGET_COL] = df_train[data.TARGET_COL].astype('int')
        # df_train[DAYSTOFIX_COL] = df_train[DAYSTOFIX_COL].astype('int')
        df_train['timestamp_fix'] = df_train['timestamp_fix'].astype('float64')
        df_train[data.FEATURES] = df_train[data.FEATURES].apply(pd.to_numeric, errors='coerce', axis=1)
        X = df_train[data.FEATURES]
        y = df_train[data.TARGET_COL]
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError('É esperada a ocorrência de duas classes para treinar.')
        self.base_learner.partial_fit(X.values, y.values, classes=[0, 1])
        for i in range(self.n_iterations): #linha 11
            # print('Na iteração {}'.format(i))
            obf0 = 1
            obf1 = 1  # linha 10
            if self.ma > self.th:  # calcular obfs (linha 15 e 16)
                obf0 = self.calcular_obf0(obf0)
            elif self.ma < self.th:
                obf1 = self.calcular_obf1(obf1)
            new_kwargs = dict(kwargs)
            new_kwargs['weights'] = [lambda0 * obf0, lambda1 * obf1]
            new_kwargs['n_iterations'] = 1
            new_kwargs['max_sample_size'] = self.max_sample_size
            # df_train[TARGET_COL] = df_train[TARGET_COL].astype('int')
            # df_train[DAYSTOFIX_COL] = df_train[DAYSTOFIX_COL].astype('int')
            # df_train['timestamp_fix'] = df_train['timestamp_fix'].astype('float64')
            # df_train[FEATURES] = df_train[FEATURES].apply(pd.to_numeric, errors='coerce', axis=1)
            if df_train.shape[0] > 0:
                s = sampler_data.df_sample_multinomial(df_train,  self.max_sample_size, [obf0, obf1], replacement=True)
                X = s[data.FEATURES]
                y = s[data.TARGET_COL]
                # treinar classificador (linha 13)
                if i == 0:
                    self.base_learner.partial_fit(X.values, y.values, classes=[0, 1])
                else:
                    self.base_learner.partial_fit(X.values, y.values)
                X_df_ma = df_ma[data.FEATURES]
                df_output = metrics.predict(self,X_df_ma) # 𝑖𝑟1 ← 1/𝑤 ∑︀𝑡 𝑗=𝑡−𝑤+1 𝑃𝑟𝑒𝑑𝑖𝑐𝑡(𝐵, ⃗𝑥𝑗, 0.5); (linha 14)

                # df_output_prediction = df_output['prediction'];
                self.ma = df_output['prediction'].mean()
                print('Self ma: {}'.format(self.ma))

            # df_test = kwargs.pop('df_val', None)
            # X_test = df_test[FEATURES]
            # y_test = df_test[TARGET_COL]
            # y_prediction = self.base_learner.predict(X_test)
            # print(classification_report(y_test, y_prediction))

    def calcular_obf1(self, obf1):
        obf1 = (((self.m ** (self.th - self.ma) - 1) * self.l1) /
                (self.m ** self.th - 1)) + 1
        return obf1

    def calcular_obf0(self, obf0):
        obf0 = ((self.m ** self.ma - self.m ** self.th) *
                self.l0) / (self.m - self.m ** self.th) + 1
        return obf0


    def save(self):
        print('salvando modelo...')
        self.classifier_train.save()

    def load(self):
        self.classifier.load()

    @property
    def n_iterations(self):
        return self.n_updates


def _track_orb(metrics, ma, lambda0, lambda1, obf0, obf1, **kwargs):
    df_val = kwargs.pop('df_val', None)
    if df_val is not None:

        print({
            'ma': ma,
            'lambda0': lambda0,
            'lambda1': lambda1,
            'obf0': obf0,
            'obf1': obf1,
        })
        return metrics
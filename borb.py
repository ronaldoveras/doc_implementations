import math

import pandas as pd


import data
import utils
import metrics as met
from borb_classifier import BORB

seed = utils.set_seed(2023)

precisions = []
recalls = []
f1_scores = []
ifas = []
pci20 = []
inspected_changes = []



# df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').apply(
#         lambda x: x.strftime('%Y/%m/%d %H:%M'))

if __name__ == '__main__':

    df = pd.read_csv('https://raw.githubusercontent.com/dinaldoap/jit-sdp-data/master/{}.csv'.format('spring-integration')
    , skipinitialspace=True)
    df = data.preprocess_daystofix(df)
    # alterarData(df)
    # separar_grupos()

    print('Estatística da base: ')
    print('Quantidade de registros {}'.format(df.shape[0]))
    # print('Quantidade de grupos separados por meses {}'.format(len(dfs)))

    th = 0.42155874754520467
    max_sample_size = 2052
    l0 = 4.535763643507429
    l1 = 2.4256762128653366
    m = 1.68225716862144
    ma_window_size = 87
    borb_waiting_time = 137
    n_updates = 18
    start = 3000
    end = None
    borb_pull_request_size = 199

    pull_request = pd.DataFrame(columns=df.columns)


    end = len(df) if end is None else end
    assert start < end, 'start deve ser menor que end.'
    interval = end - start
    n_folds = math.ceil(interval / borb_pull_request_size)  # number of folds rounded up
    end = start + n_folds * borb_pull_request_size  # last fold end
    step = borb_pull_request_size
    target_prediction = None
    for current in range(start, end, step):
        df_train = df[:current].copy()
        df_test = df[current:min(current + step, end)].copy()


        ## Treinamento com BORB
        classifier = BORB(max_sample_size=max_sample_size,
                              th=th,
                              l0=l0,
                              l1=l1,
                              m=m,
                              n_updates=n_updates)


        df_train, df_tail = data.__prepare_tail_data(df_train, ma_window_size) # pega os mais recente e coloca em df_tail

        df_train = data.prepare_train_data(
                df_train, borb_waiting_time) # contempla o algoritmo até a linha 8

        df_train = df_train.drop(data.TIMESTAMP_COL, axis=1)
        # df_train = df_train.drop('timestamp_aux', axis=1)
        df_train[data.TARGET_COL] = df_train[data.TARGET_COL].astype(data.INT_TYPE)

        df_test = df_test.drop(data.TIMESTAMP_COL, axis=1)
        # df_test = df_test.drop('timestamp_aux', axis=1)
        df_test = df_test.drop('timestamp_fix', axis=1)

        # df_test = df_test.drop(TARGET_COL, axis=1)
        classifier.train(df_train, df_ma=df_tail, df_val=None)
        print('Treinando...')
        target_prediction_test = met.predict(classifier,
            df_test)
        target_prediction = pd.concat(
            [target_prediction, target_prediction_test])
    target_prediction = target_prediction.reset_index(drop=True)
    metrics = met.prequential_recalls(target_prediction, .99)
    metrics = met.prequential_recalls_difference(metrics)
    metrics = met.prequential_gmean(metrics)
    # # metrics
    # metrics = ['r0', 'r1', 'r0-r1', 'g-mean',
    #            'tr1', 'te1', 'pr1', 'th-ma', 'th-pr1']
    # metrics
    # metrics = ['r0', 'r1', 'r0-r1']
    print('r0-r1 {}'.format(metrics['r0-r1'].mean()))
    print('g-mean {}'.format(metrics['g-mean'].mean()))

    # metrics = {metric: results[metric].mean() for metric in metrics}
            # pull_request = pd.DataFrame(columns=df.columns)
        # else:
        #     # pull_request = pd.concat([pull_request,i])
        #     pull_request.loc[len(pull_request.index) + 1] = i
            # p = (l.predict(x_test.values) >= 0.5).astype(int)
            #
            # test[PREDICTION_COL] = p
            #
            # scores = []
            # efforts = []
            # for _, row in test.iterrows():
            #     lt = row[LT_COL] * row[NF_COL]
            #     if lt == 0:
            #         lt = lt + 1
            #     effort = float((row[LA_COL] + row[LD_COL]) * lt / 2)
            #     efforts.append(effort)
            #     if row[PREDICTION_COL] == 1:
            #         scores.append(1/(effort + 1.000000001))
            #     else:
            #         scores.append(0)
            # test['density'] = scores
            # test[RISK_SCORE_COL] = scores
            # test[EFFORT_COL] = efforts
            # evaluate(test,.2)


    # print('Mediana do recall {}'.format(get_median(recalls)))
    # print('Mediana do precision {}'.format(get_median(precisions)))
    # print('Mediana do f1-score {}'.format(get_median(f1_scores)))
    # print('Mediana do IFA {}'.format(get_median(ifas)))
    # print('Mediana do PCI@20 {}'.format(get_median(pci20)))
    # print('Mediana das mudanças inspecionadas {}'.format(get_median(inspected_changes)))






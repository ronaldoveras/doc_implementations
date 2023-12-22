import math

import pandas as pd


import data
import utils
import metrics as met
from borb_classifier import BORB

seed = utils.set_seed(126553321124052187622850717793961581415)

precisions = []
recalls = []
f1_scores = []
ifas = []
pci20 = []
inspected_changes = []


if __name__ == '__main__':

    df = pd.read_csv('https://raw.githubusercontent.com/dinaldoap/jit-sdp-data/master/{}.csv'.format('nova')
    , skipinitialspace=True)
    df = data.preprocess_daystofix(df)
    # alterarData(df)
    # separar_grupos()

    print('Estatística da base: ')
    print('Quantidade de registros {}'.format(df.shape[0]))
    # print('Quantidade de grupos separados por meses {}'.format(len(dfs)))



    ## PARA SPRING-INTEGRATION
    # l0=4.719720292012793
    # l1=7.281373967241669
    # m=2.658384155903409
    # ma_window_size=193
    # borb_pull_request_size=101
    # max_sample_size=2608
    # th=0.4083754652435285
    # borb_waiting_time=92
    # n_updates = 18
    # start = 0
    # end = None
    ## PARA NOVA
    th = 0.42155874754520467
    max_sample_size = 2052
    l0 = 4.535763643507429
    l1 = 2.4256762128653366
    m = 1.68225716862144
    ma_window_size = 87
    borb_waiting_time = 137
    borb_pull_request_size = 199
    n_updates = 18
    start = 0
    end = None

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
        try:
            classifier.train(df_train, df_ma=df_tail, df_val=None)
            print('Treinando...')
        except:
            continue
        target_prediction_test = met.predict(classifier,
            df_test)
        target_prediction = pd.concat(
            [target_prediction, target_prediction_test])
    target_prediction = target_prediction.reset_index(drop=True)
    metrics = met.prequential_recalls(target_prediction, .99)
    metrics = met.prequential_recalls_difference(metrics)
    metrics = met.prequential_gmean(metrics)

    print('r0-r1 {}'.format(metrics['r0-r1'].mean()))
    print('g-mean {}'.format(metrics['g-mean'].mean()))








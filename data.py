import pandas as pd
import numpy as np

INT_TYPE = 'int'

TARGET_COL = 'target'

DAYSTOFIX_COL = 'daystofix'

NF_COL = 'nf'

LT_COL = 'lt'

LD_COL = 'ld'

LA_COL = 'la'

CONTAINS_BUG_COL = 'containsbug'

FIX_COL = 'fix'

TIMESTAMP_COL = 'timestamp'

RISK_SCORE_COL = 'risk_score'

PREDICTION_COL = 'prediction'

EFFORT_COL = 'effort'

FEATURES = [FIX_COL, 'ns', 'nd', NF_COL, 'entrophy', LA_COL,
            LD_COL, LT_COL, 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']


# [FIX_COL, 'ns', 'nd', NF_COL, 'entrophy', LT_COL, 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']
LABEL = CONTAINS_BUG_COL

def preprocess_daystofix(df_raw):
    label = CONTAINS_BUG_COL
    preprocess_cols = [TIMESTAMP_COL,
                       DAYSTOFIX_COL] + FEATURES + [label]
    df_preprocess = df_raw[preprocess_cols].copy()
    # timestamp
    df_preprocess = df_preprocess.rename(columns={label: TARGET_COL})
    # convert fix
    df_preprocess[FIX_COL] = df_preprocess[FIX_COL].astype(INT_TYPE)

    # convert contains_bug
    df_preprocess[TARGET_COL] = df_preprocess[TARGET_COL].astype(INT_TYPE)
    # timestamp fix
    bug = df_preprocess[TARGET_COL] == 1
    # df_preprocess['timestamp_aux'] = df_preprocess[TIMESTAMP_COL].astype('datetime64[ns]').apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
    df_preprocess.loc[bug, 'timestamp_fix'] = df_preprocess.loc[bug,
                                                                'timestamp'] + df_preprocess.loc[bug, DAYSTOFIX_COL] * 24 * 60 * 60

    return df_preprocess

def alterarData(df):
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], unit='s')

def separar_grupos(df):
    global dfs
    # groupby your key and freq
    g = df.groupby(pd.Grouper(key=TIMESTAMP_COL, freq='M'))
    # groups to a list of dataframes with list comprehension
    dfs = [group for _, group in g]


def __prepare_tail_data(df_train, ma_window_size):
    # most recent commits  (labeled or not)
    tail_size = min(len(df_train), ma_window_size)
    df_tail = df_train[-tail_size:]
    return df_train, df_tail

def __concat_others(df_train, train_timestamp):

    # empty df with same schema
    df_others = df_train.head(0).copy()

    df_train_others = df_others[df_others['timestamp']##aux
                                <= train_timestamp].copy()
    df_train = pd.concat([df_train, df_train_others])
    return df_train

def prepare_train_data(df_train, borb_waiting_time):
    train_timestamp = df_train['timestamp'].max() ## aux
    df_train = __concat_others(df_train, train_timestamp)

    seconds_by_day = 24 * 60 * 60
    # seconds
    verification_latency = borb_waiting_time * seconds_by_day
    # add invalid label as a safe-guard
    df_train['soft_target'] = -1.
    if df_train.empty:
        return df_train
    # check if fix has been done (bug) or verification latency has passed (normal), otherwise is unlabeled
    indices_1 = df_train['timestamp_fix'] <= train_timestamp
    indices_0 = ~indices_1 & (
        df_train['timestamp'] <= train_timestamp - verification_latency) ##aux
    indices_vl = ~indices_1 & ~indices_0

    # print('indice que não serão usados no treinamento {}'.format(indices_vl))
    df_train.loc[indices_1, 'soft_target'] = 1.
    df_train.loc[indices_0, 'soft_target'] = 0.
    # if config['uncertainty']:
    #     df_train.loc[indices_vl, 'soft_target'] = df_train[indices_vl].apply(lambda row: __verification_latency_label(
    #         train_timestamp, row.timestamp, verification_latency), axis='columns')
    # else:
    df_train.loc[indices_vl, 'soft_target'] = np.nan

    df_train = df_train.dropna(subset=['soft_target'])
    df_train[TARGET_COL] = df_train['soft_target'] > .5
    # Adicionado para effort-aware
    # print(df_train.columns)
    # df_train = df_train.drop(LA_COL,  axis='columns')
    # df_train = df_train.drop(LD_COL,  axis='columns')
    return df_train

# def change_types_boolean():
#     train[FIX_COL] = train[FIX_COL].astype(INT_TYPE)
#     test[FIX_COL] = test[FIX_COL].astype(INT_TYPE)
#     train[CONTAINS_BUG_COL] = train[CONTAINS_BUG_COL].astype(INT_TYPE)
#     test[CONTAINS_BUG_COL] = test[CONTAINS_BUG_COL].astype(INT_TYPE)
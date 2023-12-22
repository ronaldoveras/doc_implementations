import warnings
import random
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

NF_COL = ' nf'

LT_COL = ' lt'

LD_COL = ' ld'

LA_COL = ' la'

CONTAINS_BUG_COL = ' containsbug'

FIX_COL = 'fix'

TIMESTAMP_COL = ' timestamp'

RISK_SCORE_COL = 'risk_score'

PREDICTION_COL = 'prediction'

EFFORT_COL = 'effort'
warnings.filterwarnings("ignore")

FEATURES = [FIX_COL, ' ns', ' nd', NF_COL, ' entrophy', LA_COL,
            LD_COL, LT_COL, ' ndev', ' age', ' nuc', ' exp', ' rexp', ' sexp']
LABEL = CONTAINS_BUG_COL

PATH = '/home/s007840944/#Pessoal/#Doutorado/#DinaldoAP/dfs_pre_processed/nova_org.csv'

seed = random.seed(2023)

precisions = []
recalls = []
f1_scores = []
ifas = []
pci20 = []
inspected_changes = []

def evaluate(changes, effortLimit):
    """
    Evaluates the effectiveness of a change prioritization strategy.

    Args:
        changes (List[Change]): The list of changes to evaluate.
        effortLimit (double): The maximum effort that can be spent on inspecting changes.

    Returns:
        Result: The evaluation results.
    """
    # Initialize counters
    inspect_defect = 0
    inspect_change = 0
    sum_loc = 0
    cur_loc = 0
    sum_defect = 0

    # Calculate total LOC and total defects
    for _,change in changes.iterrows():
        sum_loc += change[EFFORT_COL]
        if change[LABEL]:
            sum_defect += 1

    # Determine effort limit in terms of LOC
    if effortLimit < 1:
        effortLimit = sum_loc * effortLimit

    # Sort changes by risk score
    sorted_changes = changes.sort_values([PREDICTION_COL, EFFORT_COL], ascending=[False, True])


    # Inspect changes until effort limit is reached or all changes with risk score > 0 are inspected
    for _,change in sorted_changes.iterrows():
        if change[RISK_SCORE_COL] == 0:
            break

        cur_loc += change[EFFORT_COL]
        inspect_change += 1

        if change[LABEL]:
            inspect_defect += 1

        # Record topk hits
        if inspect_defect == 1:
            ifas.append(inspect_change)

        if cur_loc >= effortLimit:
            break

    # # Set MSC if not already set
    # if result.get_msc() == 0:
    #     result.set_msc(inspect_change)

    # Calculate precision, recall, CIR, and F1-score
    if inspect_change == 0:
        precision = 0
    else:
        precision = inspect_defect / inspect_change
    precisions.append(precision)
    if sum_defect == 0:
        recall = 0
    else:
        recall = inspect_defect / sum_defect
    recalls.append(recall)
    pci20.append(inspect_change / len(changes))

    if recall + precision == 0:
        f1_scores.append(0)
    else:
        f1 = 2 * recall * precision / (recall + precision)
        f1_scores.append(f1)


    inspected_changes.append(inspect_change)



# df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').apply(
#         lambda x: x.strftime('%Y/%m/%d %H:%M'))
def alterarData(df):
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], unit='s')

def salvar(df, path):
        df.to_csv(path)

def separar_grupos():
    global dfs
    # groupby your key and freq
    g = df.groupby(pd.Grouper(key=TIMESTAMP_COL, freq='M'))
    # groups to a list of dataframes with list comprehension
    dfs = [group for _, group in g]


def change_types_boolean():
    train[FIX_COL] = train[FIX_COL].astype('int')
    test[FIX_COL] = test[FIX_COL].astype('int')
    train[CONTAINS_BUG_COL] = train[CONTAINS_BUG_COL].astype('int')
    test[CONTAINS_BUG_COL] = test[CONTAINS_BUG_COL].astype('int')


def get_median(results):
    results.sort()
    return (results[int(len(results) / 2)] + results[int((len(results) - 1) / 2)]) / 2

df = pd.read_csv(PATH, delimiter=",")
alterarData(df)
separar_grupos()

print('Estatística da base: ')
print('Quantidade de registros {}'.format(df.shape[0]))
print('Quantidade de grupos separados por meses {}'.format(len(dfs)))


i = 4
while i < len(dfs) -1:
        train = pd.concat([dfs[i-4],dfs[i-3]])
        test = pd.concat([dfs[i],dfs[i+1]])
        i = i + 1
        # print('Chegou com treinamento {}'.format(len(train)))
        # print('Chegou com teste {}'.format(len(test)))
        change_types_boolean()
        x_train = train[FEATURES]
        y_train = train[LABEL]

        # define undersample strategy
        undersample = RandomUnderSampler(sampling_strategy='majority', random_state=seed)
        # fit and apply the transform
        x_over, y_over = undersample.fit_resample(x_train.values, y_train.values)
        # print('Para treinamento {}'.format(len(x_over)))
        x_test = test[FEATURES]
        y_test = test[LABEL].astype('int')

        l = LogisticRegression(random_state=seed)
        l.fit(x_over, y_over)

        p = (l.predict(x_test.values) >= 0.5).astype(int)

        test[PREDICTION_COL] = p

        scores = []
        efforts = []
        for _, row in test.iterrows():
            lt = row[LT_COL] * row[NF_COL]
            if lt == 0:
                lt = lt + 1
            effort = float((row[LA_COL] + row[LD_COL]) * lt / 2)
            efforts.append(effort)
            if row[PREDICTION_COL] == 1:
                scores.append(1/(effort + 1.000000001))
            else:
                scores.append(0)
        test['density'] = scores
        test[RISK_SCORE_COL] = scores
        test[EFFORT_COL] = efforts
        evaluate(test,.2)


print('Mediana do recall {}'.format(get_median(recalls)))
print('Mediana do precision {}'.format(get_median(precisions)))
print('Mediana do f1-score {}'.format(get_median(f1_scores)))
print('Mediana do IFA {}'.format(get_median(ifas)))
print('Mediana do PCI@20 {}'.format(get_median(pci20)))
print('Mediana das mudanças inspecionadas {}'.format(get_median(inspected_changes)))






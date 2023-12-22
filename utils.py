import numpy as np
def random_state_seed(seed):
    return seed & (2**32 - 1)

def set_seed(seed):
    print('Seed: {}'.format(seed))
    np.random.seed(random_state_seed(seed))

def salvar(df, path):
    df.to_csv(path)


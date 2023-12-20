import collections
import pandas as pd
import torch

import data


def _sample_by_class_stratified(df, class_column, n_samples_per_class, replacement=False):
  sampled_dfs = []
  for class_, n_samples in n_samples_per_class.items():
    dados = df[df[class_column] == class_]
    if dados.shape[0] > 0:
        sampled_df = dados.sample(n_samples, replace=replacement)
        sampled_dfs.append(sampled_df)
  return pd.concat(sampled_dfs)

def df_sample_multinomial(df, k=10, probabilities=None, replacement=False):
  """
  Generates a random sample of size k from a DataFrame using multinomial sampling.

  Args:
      df: Pandas DataFrame containing the data.
      k: Number of samples to draw (default: 10).
      probabilities: Optional weights for multinomial sampling (default: uniform).
      replacement: Whether to allow replacement during sampling (default: False).

  Returns:
      A new Pandas DataFrame containing the sampled rows.
  """

  try:
    # Get data as tensor
    dados = torch.as_tensor(df.values)
  except:
      print(df)
  # Get probabilities
  if probabilities is None:
    # Uniform probabilities
    probabilities = torch.ones(dados.shape[1])
  else:
    # Ensure probabilities are normalized
    probabilities = torch.tensor(probabilities).float()
    # probabilities /= probabilities.sum()

  # Get multinomial indices
  indices = torch.multinomial(probabilities, num_samples=k, replacement=replacement)
  l = indices.tolist()

  n_samples_per_class = dict(collections.Counter(l))
  # Select rows from DataFrame using indices
  # sampled_df = df.iloc[indices.tolist()]
  sampled_df = _sample_by_class_stratified(df, data.TARGET_COL, n_samples_per_class, replacement=True)
  return sampled_df
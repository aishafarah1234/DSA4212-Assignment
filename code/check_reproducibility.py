import pandas as pd

a = pd.read_csv("results/experiment3/individual_runs.csv")
b = pd.read_csv("results/experiment3_diagnostic/individual_runs.csv")
m = a.merge(b, on=["cell", "acquisition", "seed"], suffixes=("_main", "_diag"))
print("matching runs:", len(m))
print("max difference in best_value:", (m.best_value_main - m.best_value_diag).abs().max())
print("runs where deep differs:", int((m.deep_main != m.deep_diag).sum()))
import pandas as pd

df = pd.read_csv("dataset_global.csv")
relations_uniques = df["relation"].unique()

print(relations_uniques)

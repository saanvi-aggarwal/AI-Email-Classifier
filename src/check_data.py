import pandas as pd

df = pd.read_csv('data/emails.csv')
print("Total emails:", len(df))
print("Columns:", df.columns.tolist())
print(df.head(3))
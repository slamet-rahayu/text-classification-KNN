import pandas as pd

df = pd.read_csv('metode/customers.csv', usecols=["gender","age","income","spending_score","profession","work_experience","family_size"], nrows=1600)

df['purchase'] = df.apply(
  lambda row: 1 if (row['age'] < 40 and row['income'] > 10000 and row['spending_score'] > 60)
  else 0,
  axis=1
)

df.to_csv('metode/labeled_customer_data.csv', index=False)
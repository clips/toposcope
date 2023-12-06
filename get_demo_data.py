import pandas as pd
import os
import numpy as np
from datasets import load_dataset

dataset = 'SetFit/20_newsgroups'
data = load_dataset(dataset)

df = data['train'].to_pandas() #11.3K rows
df = df.drop_duplicates(subset=['text'])
df['timestamp'] = ['2021-01-01']*3000 + ['2022-01-01']*3000 + ['2023-01-01']*3000 + ['2024-01-01']*(len(df)-9000)
df['timestamp'] = pd.to_datetime(df['timestamp'])

if not os.path.exists('demo') and not os.path.isdir('demo'):
    os.mkdir('demo')
df.to_csv('demo/demo_data.csv', index=False)

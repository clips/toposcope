import pandas as pd
import os
from datasets import load_dataset

dataset = 'SetFit/20_newsgroups'
data = load_dataset(dataset)

df = data['train'].to_pandas() #11.3K rows
df = df.drop_duplicates(subset=['text'])

if not os.path.exists('demo') and not os.path.isdir('demo'):
    os.mkdir('demo')
df.to_csv('demo/demo_data.csv', index=False)

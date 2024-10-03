import os
import pandas as pd

# get list of all files in the directory
files = os.listdir('data/initial_dataset')

# load the data from csv files
data = {}
for file in files:
    if file.endswith('.csv'):
        data[file.split('.')[0]] = pd.read_csv('data/initial_dataset/' + file)


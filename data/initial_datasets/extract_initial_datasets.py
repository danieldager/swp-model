import os, sys
import pandas as pd

# get list of all files in the directory
files = os.listdir('data/initial_datasets')

# load the data from csv files
data = {}
for file in files:
    if file.endswith('.csv'):
        data[file.split('.')[0]] = pd.read_csv('data/initial_datasets/' + file)


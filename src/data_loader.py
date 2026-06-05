# data_loader.py

import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    return df
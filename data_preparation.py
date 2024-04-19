# Filename: data_preparation.py
# This script loads the Boston housing data, performs cleaning, adds new features, and saves the processed data.
import pandas as pd
from sklearn.datasets import load_boston
import numpy as np

def prepare_data():
    # Load data
    
    data = pd.read_csv("BostonHousing.csv")
    

    # Feature engineering
    data['LSTAT_SQ'] = data['LSTAT'] ** 2
    data['LOG_PRICE'] = np.log(data['PRICE'])

    # Save processed data
    data.to_csv('processed_boston_data.csv', index=False)

if __name__ == "__main__":
    prepare_data()

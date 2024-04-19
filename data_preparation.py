import pandas as pd
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

import os, sys
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())
from ml_earthquake.collect_data import collect_data
from ml_earthquake.preprocess import _distance

def scatter(lat, lng, r):
    data_path = 'data/earthquakes_all.csv'
    if not os.path.exists(data_path):
        collect_data(data_path, start_time='1900-01-01 00:00:00')
    x_out = []
    y_out = []
    x_in = []
    y_in = []
    df = pd.read_csv(data_path, parse_dates=['time'], index_col=0)
    for date, row in tqdm(df.iterrows(), total=len(df)):
        if _distance(lat, lng, row['latitude'], row['longitude']) < r:
            x_in.append(date)
            y_in.append(row['mag'])
        else:
            x_out.append(date)
            y_out.append(row['mag'])
            
    _, ax = plt.subplots()
    ax.set_title('All Earthquakes')
    ax.set_xlabel('date')
    ax.set_ylabel('magnitude')
    ax.scatter(x_out, y_out, s=0.1, alpha=0.1)
    ax.scatter(x_in, y_in, s=0.1, label='Around Tokyo 150km')
    ax.legend(loc='upper left')

    plt.savefig('img/frequency.png')


if __name__ == '__main__':
    scatter(
        35.680934,
        139.767551,
        150 * 1000,
    )

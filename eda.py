import os
import matplotlib.pyplot as plt
from ml_earthquake.collect_data import collect_data
from ml_earthquake.preprocess import _distance, _midnight
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

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

def count(lat, lng, r, mag):
    p = 0
    n = 0
    df = pd.read_csv('data/earthquakes.csv', parse_dates=['time'], index_col=0)
    df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Asia/Tokyo')
    df = df[df['type'] == 'earthquake']
    date = _midnight(df.index.min())
    day = timedelta(days=1)
    positive = False
    for t, row in tqdm(df.iterrows(), total=len(df)):
        if date + day < t:
            date += day
            if positive: 
                p += 1
            else:
                n += 1
            positive = False
        if not positive and\
            _distance(lat, lng, row['latitude'], row['longitude']) < r and \
            mag <= row['mag']:
                positive = True
    print('P:N={}:{}'.format(p, n))

if __name__ == '__main__':
    scatter(
        35.680934,
        139.767551,
        150 * 1000,
    )
    # count(
    #     35.680934,
    #     139.767551,
    #     150 * 1000,
    #     4.0
    # )
import os, sys
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

sys.path.append(os.getcwd())
from ml_earthquake.preprocess import _distance, _midnight

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
    count(
        35.680934,
        139.767551,
        150 * 1000,
        4.0
    )

import sys, os, csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from ml_earthquake.preprocess import _midnight, _distance

class _PredictionArea:
    def __init__(self, name, center_lat, center_lng, radius_meters, threshold_mag) -> None:
        self.name = name
        self.center_lat = center_lat
        self.center_lng = center_lng
        self.radius_meters = radius_meters
        self.threshold_mag = threshold_mag

def swarm_table(areas: 'list[_PredictionArea]'):
    swarms_df = pd.read_csv('data/swarms.csv', parse_dates=['date'])
    location_ids = sorted(swarms_df['location id'].unique())
    with open('data/swarms_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(location_ids + [area.name for area in areas])

        date = None
        progress = None
        ys = np.zeros(len(areas))
        for df in pd.read_csv('data/earthquakes.csv', parse_dates=['time'], index_col=0, chunksize=1000):
            if df.index[0].tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Asia/Tokyo')
            df = df[df['type'] == 'earthquake']
            df = df.sort_index()

            if date is None:
                # first line
                date = _midnight(df.index.min())
                progress = tqdm(total=int((datetime.now(date.tzinfo) - date) // timedelta(days=1)))
            
            for d, row in df.iterrows():
                if date.day != d.day:
                    # came to next (or later) day
                    xs = np.zeros(len(location_ids))
                    swarms = swarms_df[swarms_df['date'] == date]
                    for _, swarm in swarms.iterrows():
                        xs[location_ids.index(swarm['location id'])] = swarm['count']
                    writer.writerow(np.concatenate([xs, ys]))

                    ys = np.zeros(len(areas))
                    for i in range(int((_midnight(d) - _midnight(date)) // timedelta(days=1))):
                        # blank days
                        date += timedelta(days=1)
                        progress.update()
                
                for i, area in enumerate(areas):
                    distance = _distance(row['latitude'], row['longitude'], area.center_lat, area.center_lng)
                    if distance <= area.radius_meters and area.threshold_mag <= row['mag']:
                        ys[i] = 1

if __name__ == '__main__':
    swarm_table([
        _PredictionArea('Tokyo 200km M4', 35.680934, 139.767551, 200000, 4),
        _PredictionArea('Tokyo 200km M5', 35.680934, 139.767551, 200000, 5),
        _PredictionArea('Tokyo 500km M4', 35.680934, 139.767551, 500000, 4),
        _PredictionArea('Tokyo 500km M5', 35.680934, 139.767551, 500000, 5),
        _PredictionArea('Tokyo 1000km M4', 35.680934, 139.767551, 1000000, 4),
        _PredictionArea('Tokyo 1000km M5', 35.680934, 139.767551, 1000000, 5)
    ])

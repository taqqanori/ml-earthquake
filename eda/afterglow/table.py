import sys, os, csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from ml_earthquake.preprocess import _midnight, _distance, _to_energy

class _PredictionArea:
    def __init__(self, name, center_lat, center_lng, radius_meters, threshold_mag) -> None:
        self.name = name
        self.center_lat = center_lat
        self.center_lng = center_lng
        self.radius_meters = radius_meters
        self.threshold_mag = threshold_mag

def afterglow_table(areas: 'list[_PredictionArea]', lat_gap=10, lng_gap=10):
    with open('data/afterglow_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        x_headers = []
        lat_indexes = int(180 / lat_gap) + (0 if 180 % lat_gap == 0 else 1)
        lng_indexes = int(360 / lng_gap) + (0 if 360 % lng_gap == 0 else 1)
        for i in range(lat_indexes):
            for j in range(lng_indexes):
                x_headers.append('{},{}'.format(-90 + lat_gap * i, -180 + lng_gap * j))
        x_headers.append('all')
        writer.writerow(x_headers + [area.name for area in areas])
        all_index = lat_indexes * lng_indexes

        date = None
        progress = None
        ys = np.zeros(len(areas))
        xs = np.zeros(len(x_headers))
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
                    writer.writerow(np.concatenate([xs, ys]))

                    ys = np.zeros(len(areas))
                    for i in range(int((_midnight(d) - _midnight(date)) // timedelta(days=1))):
                        # blank days
                        next_date = date + timedelta(days=1)
                        xs = xs * 0.95
                        date = next_date
                        progress.update()
                xs[lat_indexes * int((row['latitude'] - (-90)) / lat_gap) + int((row['longitude'] - (-180)) / lng_gap)] += _to_energy(row['mag'])
                xs[all_index] += _to_energy(row['mag'])
                for i, area in enumerate(areas):
                    distance = _distance(row['latitude'], row['longitude'], area.center_lat, area.center_lng)
                    if distance <= area.radius_meters and area.threshold_mag <= row['mag']:
                        ys[i] = 1
        
        # normalize
        df = pd.read_csv('data/afterglow_table.csv')
        all = df['all']
        df['all'] = (all - all.min()) / (all.max() - all.min())
        others = df[x_headers[0:-1]]
        df[x_headers[0:-1]] = (others - others.values.min()) / (others.values.max() - others.values.min())
        df.to_csv('data/afterglow_table.csv', index=False)

if __name__ == '__main__':
    afterglow_table([
        _PredictionArea('Tokyo 200km M4', 35.680934, 139.767551, 200000, 4),
        _PredictionArea('Tokyo 200km M5', 35.680934, 139.767551, 200000, 5),
        _PredictionArea('Tokyo 500km M4', 35.680934, 139.767551, 500000, 4),
        _PredictionArea('Tokyo 500km M5', 35.680934, 139.767551, 500000, 5),
        _PredictionArea('Tokyo 1000km M4', 35.680934, 139.767551, 1000000, 4),
        _PredictionArea('Tokyo 1000km M5', 35.680934, 139.767551, 1000000, 5)
    ])

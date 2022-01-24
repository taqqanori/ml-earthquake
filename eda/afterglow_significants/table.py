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

class _SignificantArea:
    next_id =  0
    def __init__(self, earthquake, lat_range, lng_range):
        self.id = _SignificantArea.next_id
        _SignificantArea.next_id += 1
        self.lat = earthquake['latitude']
        self.lng = earthquake['longitude']
        self.lat_range = lat_range
        self.lng_range = lng_range
        self.totalCount = 0

    def merge(self, earthquake):
        self.totalCount += 1
        self.lat = ((self.totalCount - 1) / self.totalCount) * self.lat + earthquake['latitude'] / self.totalCount
        self.lng = ((self.totalCount - 1) / self.totalCount) * self.lng + earthquake['longitude'] / self.totalCount

    def is_same_location(self, earthquake) -> bool:
        return abs(self.lat - earthquake['latitude']) <= self.lat_range and abs(self.lng - earthquake['longitude']) <= self.lng_range

def afterglow_significant_table(areas: 'list[_PredictionArea]', lat_range=5, lng_range=5, min_mag=6.0):
    df = pd.read_csv('data/earthquakes.csv')
    df = df[min_mag <= df['mag']]
    significant_areas: list[_SignificantArea] = []
    for _, row in df.iterrows():
        for sa in significant_areas:
            if sa.is_same_location(row):
                sa.merge(row)
                break
        else:
            significant_areas.append(_SignificantArea(row, lat_range, lng_range))

    with open('data/afterglow_significants_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        significant_area_ids = [sa.id for sa in significant_areas]
        writer.writerow(significant_area_ids + ['all'] + [area.name for area in areas])
        all_index = len(significant_area_ids)

        date = None
        progress = None
        ys = np.zeros(len(areas))
        xs = np.zeros(len(significant_areas) + 1)
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

                for sa in significant_areas:
                    if sa.is_same_location(row):
                        xs[significant_area_ids.index(sa.id)] += _to_energy(row['mag'])
                        break
                xs[all_index] += _to_energy(row['mag'])

                for i, area in enumerate(areas):
                    distance = _distance(row['latitude'], row['longitude'], area.center_lat, area.center_lng)
                    if distance <= area.radius_meters and area.threshold_mag <= row['mag']:
                        ys[i] = 1
        
        # normalize
        df = pd.read_csv('data/afterglow_significants_table.csv')
        all = df['all']
        df['all'] = (all - all.min()) / (all.max() - all.min())
        significant_area_ids_str = [str(id) for id in significant_area_ids]
        x_df = df[significant_area_ids_str]
        df[significant_area_ids_str] = (x_df - x_df.values.min()) / (x_df.values.max() - x_df.values.min())
        df.to_csv('data/afterglow_significants_table.csv', index=False)

if __name__ == '__main__':
    afterglow_significant_table([
        _PredictionArea('Tokyo 200km M4', 35.680934, 139.767551, 200000, 4),
        _PredictionArea('Tokyo 200km M5', 35.680934, 139.767551, 200000, 5),
        _PredictionArea('Tokyo 500km M4', 35.680934, 139.767551, 500000, 4),
        _PredictionArea('Tokyo 500km M5', 35.680934, 139.767551, 500000, 5),
        _PredictionArea('Tokyo 1000km M4', 35.680934, 139.767551, 1000000, 4),
        _PredictionArea('Tokyo 1000km M5', 35.680934, 139.767551, 1000000, 5)
    ])

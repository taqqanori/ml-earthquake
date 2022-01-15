import csv
import pandas as pd
from tqdm import tqdm

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

def significants(lat_range=1, lng_range=1, min_mag=6.0):
    df = pd.read_csv('data/earthquakes.csv', parse_dates=['time'])
    df = df[df['type'] == 'earthquake']
    df['time'] = df['time'].dt.tz_convert('Asia/Tokyo')
    df = df[min_mag <= df['mag']]
    areas: list[_SignificantArea] = []
    with open('data/significants.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['location id', 'date', 'latitude', 'longitude', 'magnitude'])
        progress = tqdm(total=len(df))
        area = None
        for _, row in df.iterrows():
            for a in areas:
                if a.is_same_location(row):
                    a.merge(row)
                    area = a
                    break
            else:
                area = _SignificantArea(row, lat_range, lng_range)
                areas.append(area)
            writer.writerow([area.id, row['time'], row['latitude'], row['longitude'], row['mag']])
            progress.update()

if __name__ == '__main__':
    significants()

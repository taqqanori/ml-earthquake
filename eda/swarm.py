import sys, os, csv
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from ml_earthquake.preprocess import _midnight

class _SwarmArea:
    next_id =  0
    def __init__(self, earthquake, date_range, count_range, lat_range, lng_range):
        self.id = None
        self.lat = earthquake['latitude']
        self.lng = earthquake['longitude']
        self.date_range = date_range
        self.count_range = count_range
        self.lat_range = lat_range
        self.lng_range = lng_range
        self.totalCount = 0
        self.earthquakes = [earthquake]

    def add(self, earthquake):
        self.totalCount += 1
        self.lat = ((self.totalCount - 1) / self.totalCount) * self.lat + earthquake['latitude'] / self.totalCount
        self.lng = ((self.totalCount - 1) / self.totalCount) * self.lng + earthquake['longitude'] / self.totalCount
        self.earthquakes.append(earthquake)

    def is_same_location(self, earthquake) -> bool:
        return abs(self.lat - earthquake['latitude']) <= self.lat_range and abs(self.lng - earthquake['longitude']) <= self.lng_range
    
    def output(self, writer):
        date = None
        buf = []
        for eq in sorted(self.earthquakes, key=lambda x: x['time']):
            if date is None:
                # first 
                date = _midnight(eq['time'])
                continue
            if date.day != eq['time'].day:
                # came to next (or later) day
                if self.count_range <= len(buf):
                    if self.id is None:
                        self.id = _SwarmArea.next_id
                        _SwarmArea.next_id += 1
                    avg = 0.0
                    for b in buf:
                        avg += b['mag']
                    avg /= len(buf)
                    writer.writerow([self.id, date, self.lat, self.lng, len(buf), avg])
                for _ in range(int((_midnight(eq['time']) - _midnight(date)) // timedelta(days=1))):
                    # blank days
                    date += timedelta(days=1)
                limit_date = date - timedelta(days=self.date_range)
                buf = [b for b in buf if limit_date <= b['time']]
            buf.append(eq)

def swarm(date_range=7, count_range=10, lat_range=1, lng_range=1):
    df = pd.read_csv('data/earthquakes.csv', parse_dates=['time'])
    df = df[df['type'] == 'earthquake']
    df['time'] = df['time'].dt.tz_convert('Asia/Tokyo')
    df = df.sort_values(['latitude', 'longitude', 'time'])
    areas: list[_SwarmArea] = []
    with open('data/swarms.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['location id', 'date', 'latitude', 'longitude', 'count', 'average magnitude'])
        progress = tqdm(total=len(df))
        for _, row in df.iterrows():
            found = False
            for area in areas:
                if area.is_same_location(row):
                    area.add(row)
                    found = True
                elif lat_range < abs(area.lat - row['latitude']):
                    area.output(writer)
                    areas.remove(area)
            if not found:
                areas.append(_SwarmArea(row, date_range, count_range, lat_range, lng_range))
            progress.update()
        for area in areas:
            area.output(writer)

if __name__ == '__main__':
    swarm()

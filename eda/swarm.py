import sys, os, csv
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from ml_earthquake.preprocess import _midnight

class Swarm:
    def __init__(self, earthquake):
        self.lat = earthquake['latitude']
        self.lng = earthquake['longitude']
        self.totalCount = 0
        self.earthquakes = [earthquake]

    def add(self, earthquake):
        self.totalCount += 1
        self.lat = ((self.totalCount - 1) / self.totalCount) * self.lat + earthquake['latitude'] / self.totalCount
        self.lng = ((self.totalCount - 1) / self.totalCount) * self.lng + earthquake['longitude'] / self.totalCount
        self.earthquakes.append(earthquake)

def swarm(date_range = 7, count_range = 10, lat_range=0.001, lng_range=0.001):
    date = None
    progress = None
    buff = []
    swarms = []
    with open('data/swarms.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'latitude', 'longitude', 'count', 'average magnitude'])
        for df in pd.read_csv('data/earthquakes.csv', parse_dates=['time'], index_col=0, chunksize=100):
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
                    _out(writer, swarms, date, date_range, count_range)
                found = False
                for swarm in swarms:
                    if abs(swarm.lat - row['latitude']) <= lat_range and abs(swarm.lng - row['longitude']) <= lng_range:
                        swarm.add(_row_with_time(d, row))
                        found = True
                for b in buff:
                    if abs(b['latitude'] - row['latitude']) <= lat_range and abs(b['longitude'] - row['longitude']) <= lng_range:
                        swarm = Swarm(_row_with_time(d, row))
                        swarm.add(b)
                        swarms.append(swarm)
                        found = True
                        break
                if not found:
                    buff.append(_row_with_time(d, row))

            for i in range(int((_midnight(d) - _midnight(date)) // timedelta(days=1))):
                # blank days
                date += timedelta(days=1)
                limit_date = date - timedelta(days=date_range)
                buff = [b for b in buff if limit_date <= b['time']]
                progress.update()

        # last day
        _out(writer, swarms, date, date_range, count_range)

def _row_with_time(time, row):
    ret = row.copy()
    ret['time'] = time
    return ret

def _out(writer, swarms, date, date_range, count_range):
    for swarm in swarms:
        if count_range <= len(swarm.earthquakes):
            avg = 0
            for eq in swarm.earthquakes:
                avg += eq['mag']
            avg /= len(swarm.earthquakes)
            writer.writerow([date, swarm.lat, swarm.lng, len(swarm.earthquakes), avg])
        limit_date = date - timedelta(date_range - 1)
        swarm.earthquakes = [eq for eq in swarm.earthquakes if limit_date <= eq['time']]

if __name__ == '__main__':
    swarm()

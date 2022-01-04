import sys, os, csv
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from ml_earthquake.preprocess import _midnight

class _TimeRowSet:
    def __init__(self, time, row):
        self.time = time
        self.row = row
class _Swarm:
    def __init__(self, earthquake: _TimeRowSet):
        self.lat = earthquake.row['latitude']
        self.lng = earthquake.row['longitude']
        self.totalCount = 0
        self.earthquakes = [earthquake]

    def add(self, earthquake: _TimeRowSet):
        self.totalCount += 1
        self.lat = ((self.totalCount - 1) / self.totalCount) * self.lat + earthquake.row['latitude'] / self.totalCount
        self.lng = ((self.totalCount - 1) / self.totalCount) * self.lng + earthquake.row['longitude'] / self.totalCount
        self.earthquakes.append(earthquake)

def swarm(date_range=7, count_range=10, lat_range=0.005, lng_range=0.005, min_mag=2.0):
    date = None
    progress = None
    buff: list[_Swarm] = []
    swarms: list[_Swarm] = []
    with open('data/swarms.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['location id', 'date', 'latitude', 'longitude', 'count', 'average magnitude'])
        for df in pd.read_csv('data/earthquakes.csv', parse_dates=['time'], index_col=0, chunksize=100):
            df = df[min_mag <= df['mag']]
            if len(df.index) <= 0:
                continue
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
                    for i in range(int((_midnight(d) - _midnight(date)) // timedelta(days=1))):
                        # blank days
                        date += timedelta(days=1)
                        progress.update()
                    limit_date = date - timedelta(days=date_range)
                    _remove_earthquakes_before(limit_date, swarms)
                    _remove_earthquakes_before(limit_date, buff)
                    buff = [b for b in buff if 0 <= len(b.earthquakes)]
                found = False
                for swarm in swarms:
                    if _is_same_location(swarm, row, lat_range, lng_range):
                        swarm.add(_TimeRowSet(d, row))
                        found = True
                        break
                if not found:
                    for b in buff:
                        if _is_same_location(b, row, lat_range, lng_range):
                            b.add(_TimeRowSet(d, row))
                            if count_range <= len(b.earthquakes):
                                swarms.append(b)
                                buff.remove(b)
                            found = True
                            break
                if not found:
                    buff.append(_Swarm(_TimeRowSet(d, row)))

        # last day
        _out(writer, swarms, date, date_range, count_range)

def _is_same_location(swarm: _Swarm, row, lat_range, lng_range) -> bool:
    return abs(swarm.lat - row['latitude']) <= lat_range and abs(swarm.lng - row['longitude']) <= lng_range

def _out(writer, swarms, date, date_range, count_range):
    for i, swarm in enumerate(swarms):
        if count_range <= len(swarm.earthquakes):
            avg = 0
            for eq in swarm.earthquakes:
                avg += eq.row['mag']
            avg /= len(swarm.earthquakes)
            writer.writerow([i, date, swarm.lat, swarm.lng, len(swarm.earthquakes), avg])

def _remove_earthquakes_before(date, swarms: 'list[_Swarm]'):
    for swarm in swarms:
        swarm.earthquakes = [eq for eq in swarm.earthquakes if date <= eq.time]

if __name__ == '__main__':
    swarm()

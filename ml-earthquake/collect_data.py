
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import pandas as pd
import os

url_base = 'https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime={}%2000:00:00&endtime={}%2023:59:59&minmagnitude={}&orderby=time-asc'

def collect_data(out_path, start_date='1945-01-01', end_date=None, min_mag=0, step_days=30, show_progress=True):
    f = '%Y-%m-%d'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    start_date = datetime.strptime(start_date + ' +0000', f + ' %z')
    if end_date is None:
        end_date = datetime.now(tz=timezone.utc)
    date = start_date
    step_days = timedelta(days=step_days)

    if show_progress:
        progress = tqdm(total=int((end_date - start_date) / step_days))
    
    df = None
    while date <= end_date:
        url = url_base.format(\
            date.strftime(f), \
            (date + step_days).strftime(f), \
            min_mag)
        step_df = pd.read_csv(url)
        if df is not None:
            df = pd.concat([df, step_df])
        else:
            df = step_df

        date += step_days
        if show_progress:
            progress.update()
    df.to_csv(out_path, index=None)

if __name__ == '__main__':
    collect_data('../data/earthquakes.csv')
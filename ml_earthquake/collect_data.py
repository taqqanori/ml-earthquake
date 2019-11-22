
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import pandas as pd
import os

url_base = 'https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime={}%2000:00:00&endtime={}%2023:59:59&minmagnitude={}&orderby=time-asc'

def collect_data(out_path, start_date='1980-01-01', end_date=None, min_mag=0, step_days=30, show_progress=True):
    f = '%Y-%m-%d'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    start_date = datetime.strptime(start_date + ' +0000', f + ' %z')
    if end_date is None:
        end_date = datetime.now(tz=timezone.utc)
    date = start_date

    if show_progress:
        progress = tqdm(total=int((end_date - start_date) / timedelta(days=step_days)))
    
    df = None
    subdivide = 1
    while date <= end_date:
        step_df = None
        for start_offset in range(0, step_days, step_days // subdivide):
            end_offset = min(start_offset + (step_days // subdivide), step_days)
            url = url_base.format(\
                (date + timedelta(days=start_offset)).strftime(f), \
                (date + timedelta(days=end_offset)).strftime(f), \
                min_mag)
            try:
                step_df = _concat(step_df, pd.read_csv(url))
            except Exception as e:
                # failure
                print(e)
                subdivide += 1
                print('retrying (dividing date range into {})...'.format(subdivide))
                break
        else:
            # success!
            subdivide = 1
            df = _concat(df, step_df)

            date += timedelta(days=step_days)
            if show_progress:
                progress.update()
    df.to_csv(out_path, index=None)

def _concat(df1, df2):
    if df1 is None:
        return df2
    return pd.concat([df1, df2])

if __name__ == '__main__':
    collect_data(
        os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'data', 
            'earthquakes.csv'))
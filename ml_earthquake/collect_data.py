
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import pandas as pd
import os

url_base = 'https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime={}&endtime={}&minmagnitude={}&orderby=time-asc'
f = '%Y-%m-%d %H:%M:%S'

# date should be specified by UTC!
def collect_data(out_path, start_time='1980-01-01 00:00:00', end_time=None, min_mag=0, step_days=30, show_progress=True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    start_time = _utc_time(start_time)
    if end_time is None:
        end_time = datetime.now(tz=timezone.utc)
    else:
        end_time = _utc_time(end_time)
    t = start_time

    if show_progress:
        progress = tqdm(total=int((end_time - start_time) / timedelta(days=step_days)))
    
    df = None
    subdivide = 1
    while t <= end_time:
        step_df = None
        for start_offset in range(0, step_days, step_days // subdivide):
            end_offset = min(start_offset + (step_days // subdivide), step_days)
            url = url_base.format(\
                (t + timedelta(days=start_offset)).strftime(f).replace(" ", "%20"), \
                (t + timedelta(days=end_offset)).strftime(f).replace(" ", "%20"), \
                min_mag)
            try:
                step_df = _concat(step_df, pd.read_csv(url, parse_dates=['time']))
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

            t += timedelta(days=step_days)
            if show_progress:
                progress.update()
    df = df.sort_values('time')
    df.to_csv(out_path, index=None)

def _utc_time(s):
    return datetime.strptime(s + ' +0000', f + ' %z')

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
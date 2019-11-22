import os, math, pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

def preprocess(
    data_path,
    window_days,
    predict_range_days,
    lat_gralularity,
    lng_gralularity,
    predict_center_lat,
    predict_center_lng,
    predict_radius_meters,
    threshold_mag,
    cache_dir=None,
    show_progress=True
):
    cache_X_path = None
    cache_y_path = None
    cache_info_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_X_path = os.path.join(cache_dir, 'X_{}_{}_{}_{}.npy'.format(
            window_days,
            lat_gralularity,
            lng_gralularity,
            predict_range_days,
        ))
        y_info_id = '{}_{}_{}_{}_{}_{}'.format(
            window_days,
            predict_range_days,
            predict_center_lat,
            predict_center_lng,
            predict_radius_meters,
            threshold_mag
        )
        cache_y_path = os.path.join(cache_dir, 'y_{}.npy'.format(y_info_id))
        cache_info_path = os.path.join(cache_dir, 'info_{}.pickle'.format(y_info_id))
    
    df = pd.read_csv(data_path, parse_dates=['time'], index_col=0)
    df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Asia/Tokyo')
    df = df[df['type'] == 'earthquake']
    date = _midnight(df.index.min())
    end_date = _midnight(datetime.now(date.tzinfo) + timedelta(days=1))
    window = timedelta(days=window_days)
    predict_range = timedelta(days=predict_range_days)
    lat_gap = 180 / lat_gralularity
    lng_gap = 360 / lng_gralularity
    progress = None
    if show_progress:
        progress = tqdm(total=int((end_date - date) / (window + predict_range)))
    X = []
    y = []
    info = []
    while date < end_date:
        # X
        if not os.path.exists(cache_X_path):
            x = np.zeros([window_days, lat_gralularity, lng_gralularity, 2])
            for win in range(window_days):
                for _, row in _range(df, date + timedelta(days=win), date + timedelta(days=win+1)).iterrows():
                    lat_index = min(int((row['latitude'] - (-90)) / lat_gap), lat_gralularity - 1)
                    lng_index = min(int((row['longitude'] - (-180)) / lng_gap), lng_gralularity - 1)
                    # ch1: magnitude
                    x[win, lat_index, lng_index, 0] = _sum_mag(row['mag'], x[win, lat_index, lng_index, 0])
                    # ch2: frequency
                    x[win, lat_index, lng_index, 1] += 1
            X.append(x)

        # y, info
        if not os.path.exists(cache_y_path) or not os.path.exists(cache_info_path):
            _y = 0
            i = {
                'window_start': date,
                'window_end': date + window,
                'predict_start': date + window,
                'predict_end': date + window + predict_range,
                'threshold_mag': threshold_mag,
                'earthquakes': []
            }
            for time, row in _range(df, date + window, date + window + predict_range).iterrows():
                d = _distance(row['latitude'], row['longitude'], predict_center_lat, predict_center_lng)
                if d <= predict_radius_meters:
                    if threshold_mag <= row['mag']:
                        _y = 1
                    i['earthquakes'].append({
                        'time': time,
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'mag': row['mag']
                    })
            y.append(_y)
            info.append(i)

        date += (window + predict_range)
        if show_progress:
            progress.update()

    if os.path.exists(cache_X_path):
        X = np.load(cache_X_path)
    else:
        X = np.array(X)
        # normalize
        for i in range(X.shape[4]):
            X[:,:,:,:,i] = X[:,:,:,:,i] / X[:,:,:,:,i].max()
        np.save(cache_X_path, X)
        
    if os.path.exists(cache_y_path):
        y = np.load(cache_y_path)
    else:
        y = np.array(y)
        np.save(cache_y_path, y)

    if os.path.exists(cache_info_path):
        with open(cache_info_path, 'rb') as f:
            info = pickle.load(f)
    else:
        with open(cache_info_path, 'wb') as f:
            pickle.dump(info, f)
    
    return X, y, info

def _midnight(d):
    return datetime(year=d.year, month=d.month, day=d.day, tzinfo=d.tzinfo)

def _range(df, start, end):
    ret = df[start <= df.index]
    ret = ret[ret.index < end]
    return ret

# log10(E) = 4.8 + 1.5M
def _sum_mag(m1, m2):
    e1 = _to_energy(m1)
    e2 = _to_energy(m2)
    return (math.log10(e1 + e2) - 4.8) / 1.5

def _to_energy(m):
    return math.pow(10, 4.8 + 1.5 * m)

def _distance(lat1, lng1, lat2, lng2):
    rad_lat1 = np.deg2rad(lat1)
    rad_lng1 = np.deg2rad(lng1)
    rad_lat2 = np.deg2rad(lat2)
    rad_lng2 = np.deg2rad(lng2)

    r = 6378137.0

    avg_lat = (rad_lat1 - rad_lat2) / 2
    avg_lng = (rad_lng1 - rad_lng2) / 2
    return r * 2 * math.asin(math.sqrt(math.pow(math.sin(avg_lat), 2) + math.cos(rad_lat1) * math.cos(rad_lat2) * math.pow(math.sin(avg_lng), 2)))

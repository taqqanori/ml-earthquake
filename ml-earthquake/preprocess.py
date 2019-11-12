import os, math
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
    cache_id = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
        window_days,
        lat_gralularity,
        lng_gralularity,
        predict_range_days,
        predict_center_lat,
        predict_center_lng,
        predict_radius_meters,
        threshold_mag
    )
    cache_X_path = None
    cache_y_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_X_path = os.path.join(cache_dir, 'X_{}.npy'.format(cache_id))
        cache_y_path = os.path.join(cache_dir, 'y_{}.npy'.format(cache_id))
        if os.path.exists(cache_X_path) and os.path.exists(cache_y_path):
            return np.load(cache_X_path), np.load(cache_y_path) 
    
    df = pd.read_csv(data_path, parse_dates=['time'])
    date = _midnight(df['time'].min())
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
    while date < end_date:
        # X
        x = np.zeros([lat_gralularity, lng_gralularity, 2])
        for _, row in _range(df, date, date + window).iterrows():
            lat_index = min(int((row['latitude'] - (-90)) / lat_gap), lat_gralularity - 1)
            lng_index = min(int((row['longitude'] - (-180)) / lng_gap), lng_gralularity - 1)
            # ch1: magnitude
            x[lat_index, lng_index, 0] = _sum_mag(row['mag'], x[lat_index, lng_index, 0])
            # ch2: frequency
            x[lat_index, lng_index, 1] += 1
        X.append(x)

        # y
        _y = 0
        for _, row in _range(df, date + window, date + window + predict_range).iterrows():
            d = _distance(row['latitude'], row['longitude'], predict_center_lat, predict_center_lng)
            if d <= predict_radius_meters and threshold_mag <= row['mag']:
                _y = 1
        y.append(_y)

        date += (window + predict_range)
        if show_progress:
            progress.update()
    
    X = np.array(X)
    y = np.array(y)

    # normalize
    X[:,:,:,0] = X[:,:,:,0] / X[:,:,:,0].max()
    X[:,:,:,1] = X[:,:,:,1] / X[:,:,:,1].max()

    if cache_dir is not None:
        np.save(cache_X_path, X)
        np.save(cache_y_path, y)
    
    return X, y

def _midnight(d):
    return datetime(year=d.year, month=d.month, day=d.day, tzinfo=d.tzinfo)

def _range(df, start, end):
    ret = df[start <= df['time']]
    ret = ret[ret['time'] < end]
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

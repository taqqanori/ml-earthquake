import os
import math
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from multiprocessing import Pool, Manager
from operator import itemgetter


def preprocess(
    data_path,
    window_days,
    predict_range_days,
    predict_center_lat,
    predict_center_lng,
    predict_radius_meters,
    threshold_mag,
    num_points=1000,
    normalize_max_mag=10.0,
    normalize_max_depth=500,
    for_prediction=False,
    test_ratio=0.25,
    cache_dir=None,
    show_progress=True,
    processes=5,
    chunk_size=100
):
    cache_X_path = None
    cache_y_path = None
    cache_info_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_X_path = os.path.join(cache_dir, 'X_{}_{}.npy'.format(
            window_days,
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
        cache_info_path = os.path.join(
            cache_dir, 'info_{}.pickle'.format(y_info_id))

        if os.path.exists(cache_X_path) and \
                os.path.exists(cache_y_path) and \
                os.path.exists(cache_info_path):
            X = np.load(cache_X_path)
            y = np.load(cache_y_path)
            with open(cache_info_path, 'rb') as f:
                info = pickle.load(f)
            return _train_test_split(X, y, info, window_days, predict_range_days, test_ratio)

    X = []
    y = []
    info = []
    X_buf = []
    y_buf = []
    eq_buf = []
    x = None
    _y = None
    eq = None
    date = None
    progress = None
    for df in pd.read_csv(data_path, parse_dates=['time'], index_col=0, chunksize=chunk_size):
        if df.index[0].tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Tokyo')
        df = df[df['type'] == 'earthquake']
        df = df.sort_index()

        if date is None:
            # first line
            date = _midnight(df.index.min())
            if show_progress:
                progress = tqdm(
                    total=int((datetime.now(date.tzinfo) - date) // timedelta(days=1)))
            x = []
            _y = False
            eq = []

        for d, row in df.iterrows():
            if date.day != d.day:
                # came to next (or later) day
                date += timedelta(days=1)
                if show_progress:
                    progress.update()
                _append(date, X, y, info, x, _y, eq, X_buf, y_buf, eq_buf,
                        predict_center_lat, predict_center_lng, window_days, predict_range_days, threshold_mag,
                        num_points, normalize_max_mag, normalize_max_depth)
                x = []
                _y = False
                eq = []
                for i in range(int((_midnight(d) - _midnight(date)) // timedelta(days=1))):
                    # blank days
                    _append(date, X, y, info, x, _y, eq, X_buf, y_buf, eq_buf,
                            predict_center_lat, predict_center_lng, window_days, predict_range_days, threshold_mag,
                            num_points, normalize_max_mag, normalize_max_depth)
                    date += timedelta(days=1)
                    if show_progress:
                        progress.update()

            # x
            x.append(row)

            # y
            distance = _distance(
                row['latitude'], row['longitude'], predict_center_lat, predict_center_lng)
            if distance <= predict_radius_meters:
                if threshold_mag <= row['mag']:
                    _y = True
                    # info
                    eq.append({
                        'time': d,
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'depth': row['depth'],
                        'mag': row['mag']
                    })

    # the last day
    _append(date, X, y, info, x, _y, eq, X_buf, y_buf, eq_buf,
            predict_center_lat, predict_center_lng, window_days, predict_range_days, threshold_mag,
            normalize_max_mag, normalize_max_depth)

    if for_prediction:
        X.append(_to_points(X_buf[0:window_days], num_points, normalize_max_mag, normalize_max_depth))

    X = np.array(X)
    # normalize

    if for_prediction:
        return X

    np.save(cache_X_path, X)

    y = np.array(y)
    np.save(cache_y_path, y)

    with open(cache_info_path, 'wb') as f:
        pickle.dump(info, f)

    return _train_test_split(X, y, info, window_days, predict_range_days, test_ratio)


def _train_test_split(X, y, info, window_days, predict_range_days, test_ratio):
    test_count = int(len(X) * test_ratio)
    train_index = np.arange(
        0, len(X) - (test_count + window_days + predict_range_days))
    test_index = np.arange(len(X) - test_count, len(X))
    info = np.array(info)
    return X[train_index], y[train_index], X[test_index], y[test_index], info[train_index], info[test_index]


def _append(
        date, X, y, info, x, _y, eq, X_buf, y_buf, eq_buf,
        predict_center_lat,
        predict_center_lng,
        window_days,
        predict_range_days,
        threshold_mag,
        num_points,
        normalize_max_mag,
        normalize_max_depth):
    X_buf.append(x)
    y_buf.append(_y)
    eq_buf.append(eq)
    if len(X_buf) < window_days + predict_range_days:
        # just beginning, not enough stored in buf
        if predict_range_days < len(y_buf):
            y_buf.pop(0)
        if predict_range_days < len(eq_buf):
            eq_buf.pop(0)
        return
    X.append(_to_points(X_buf[0:window_days], num_points, normalize_max_mag, normalize_max_depth))
    X_buf.pop(0)
    y_buf.pop(0)
    eq_buf.pop(0)
    y.append(any(y_buf))
    info.append({
        'predict_center_lat': predict_center_lat,
        'predict_center_lng': predict_center_lng,
        'window_start': date - timedelta(days=window_days + predict_range_days),
        'window_end': date - timedelta(days=predict_range_days),
        'predict_start': date - timedelta(days=predict_range_days),
        'predict_end': date,
        'threshold_mag': threshold_mag,
        'earthquakes': sum(eq_buf, [])
    })


def _to_points(buf, num_points, normalize_max_mag, normalize_max_depth):
    all_points = []
    range_days = len(buf)
    for i, buf_day in enumerate(buf):
        for row in buf_day:
            all_points.append([
                (row['latitude'] - (-90)) / 180,
                (row['longitude'] - (-180)) / 360,
                max(0, min(1, row['depth'] / normalize_max_depth)),
                min(1, row['mag'] / normalize_max_mag),
                (range_days - i) / range_days
            ])
    if len(all_points) < num_points:
        raise "points are less than " + num_points + "(" + len(all_points) + ")"
    all_points.sort(reverse=True, key=itemgetter(3))
    return np.array(all_points[0:num_points])


def _midnight(d):
    return datetime(year=d.year, month=d.month, day=d.day, tzinfo=d.tzinfo)


def _range(df, start, end):
    ret = df[start <= df.index]
    ret = ret[ret.index < end]
    return ret

# log10(E) = 4.8 + 1.5M


def _sum_mag(m1, m2):
    if m1 == 0:
        return m2
    if m2 == 0:
        return m1
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

import fire, os, json, sys
from datetime import datetime, timezone, timedelta
from dateutil.tz import gettz
from keras.models import load_model

sys.path.append(os.getcwd())
from ml_earthquake import collect_data
from ml_earthquake import preprocess

date_format = '%Y%m%d'
file_date_format = '%Y-%m-%d'
datetime_format = '%Y-%m-%d %H:%M:%S'
file_datetime_format = '%Y%m%d_%H%M%S'


def main(
        recipe='recipe.json',
        recipe_id=None,
        target_date=None,
        tz='Asia/Tokyo',
        out_dir='out',
        work_dir='work'):

    if not os.path.exists(recipe):
        raise Exception('recipe: {} does not exists...'.format(recipe))

    tz = gettz(tz)
    target_date = _midnight(datetime.now(tz=tz)) if target_date is None \
        else _midnight(datetime.strptime(target_date, '%Y-%m-%d')).astimezone(tz)
    window_end = target_date.astimezone(timezone.utc)

    out = {
        'predictions': []
    }
    with open(recipe, 'r', encoding='utf-8') as f:
        recipe_obj = json.load(f)
        for r in recipe_obj['recipe']:
            if recipe_id is not None and recipe_id != r['id']:
                continue
            window_start = window_end - timedelta(days=r['window_days'])
            predict_start = window_end.astimezone(tz).strftime(date_format)
            predict_end = (window_end + timedelta(days=r['predict_range_days']))\
                .astimezone(tz)\
                .strftime(date_format)
            csv_path = os.path.join(work_dir,
                                    'earthquakes_{}_{}.csv'.format(
                                        window_start.strftime(
                                            file_datetime_format),
                                        window_end.strftime(file_datetime_format)))
            if not os.path.exists(csv_path):
                collect_data(
                    out_path=csv_path,
                    start_time=window_start.strftime(datetime_format),
                    end_time=window_end.strftime(datetime_format),
                    show_progress=False)
            X = preprocess(
                csv_path,
                r['window_days'],
                r['predict_range_days'],
                r['predict_center_lat'],
                r['predict_center_lng'],
                r['predict_radius_meters'],
                r['threshold_mag'],
                cache_dir=None,
                show_progress=False,
                for_prediction=True
            )
            model = load_model(os.path.join(out_dir, r['id'], 'best_model.h5'))
            y = model.predict(X)[0][0]
            with open(os.path.join(out_dir, r['id'], 'summary.json'), 'r', encoding='utf-8') as s:
                summary = json.load(s)
                out['predictions'].append({
                    'id': r['id'],
                    'start_date': predict_start,
                    'predict_center_lat': r['predict_center_lat'],
                    'predict_center_lng': r['predict_center_lng'],
                    'predict_radius_meters': r['predict_radius_meters'],
                    'predict_range_days': r['predict_range_days'],
                    'probability': float(y),
                    'displayName': r['displayName'],
                    'threshold_mag': r['threshold_mag'],
                    'acc': summary['acc'],
                    'auc': summary['auc'],
                    'f1': summary['f1'],
                    'precision': summary['precision'],
                    'recall': summary['recall'],
                })

            # detail
            detail = {
                'mag_heatmaps': [],
                'freq_heatmaps': [],
                'depth_heatmaps': [],
                'lat_gap': 180 / X[0].shape[1],
                'lng_gap': 360 / X[0].shape[2],
                'threshold_mag': r['threshold_mag'],
                'earthquakes': []
            }
            for x in X[0]:
                mag_heatmap = []
                freq_heatmap = []
                depth_heatmap = []
                for lat in range(x.shape[0]):
                    for lng in range(x.shape[1]):
                        mag = x[lat][lng][0]
                        freq = x[lat][lng][1]
                        depth = x[lat][lng][2]
                        if 0 < mag:
                            mag_heatmap.append({
                                'lat': lat,
                                'lng': lng,
                                'heat': mag
                            })
                        if 0 < freq:
                            freq_heatmap.append({
                                'lat': lat,
                                'lng': lng,
                                'heat': freq
                            })
                        if 0 < depth:
                            depth_heatmap.append({
                                'lat': lat,
                                'lng': lng,
                                'heat': depth
                            })
                detail['mag_heatmaps'].append(mag_heatmap)
                detail['freq_heatmaps'].append(freq_heatmap)
                detail['depth_heatmaps'].append(depth_heatmap)
            detail_path = 'detail_{}_{}.json'.format(
                predict_start, predict_end)
            with open(os.path.join(out_dir, r['id'], detail_path), 'w', encoding='utf-8') as d:
                json.dump(detail, d, ensure_ascii=False)

    with open(os.path.join(out_dir, 'predictions.json'), 'w', encoding='utf-8') as o:
        json.dump(out, o, ensure_ascii=False)


def _midnight(d):
    return datetime(year=d.year, month=d.month, day=d.day, tzinfo=d.tzinfo)


if __name__ == '__main__':
    fire.Fire(main)

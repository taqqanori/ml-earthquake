import fire, os, json, sys
from datetime import datetime, timezone, timedelta
from dateutil.tz import gettz
from keras.models import load_model

sys.path.append(os.getcwd())
from ml_earthquake import collect_data
from ml_earthquake import preprocess

date_format = '%Y-%m-%d %H:%M:%S'
file_date_format = '%Y%m%d_%H%M%S'

def main(recipe='recipe.json', recipe_id=None, out_dir='out', work_dir='work', tz='Asia/Tokyo'):

    if not os.path.exists(recipe):
        raise Exception('recipe: {} does not exists...'.format(recipe))

    tz = gettz(tz)
    window_end = (_midnight(datetime.now(tz=tz)) + timedelta(days=1)).astimezone(timezone.utc)

    out = {
        'predictions': []
    }
    with open(recipe, 'r', encoding='utf-8') as f:
        recipe_obj = json.load(f)
        for r in recipe_obj['recipe']:
            if recipe_id is not None and recipe_id != r['id']:
                continue
            window_start = window_end - timedelta(days=r['window_days'])
            csv_path = os.path.join(work_dir,\
                'earthquakes_{}_{}.csv'.format(\
                    window_start.strftime(file_date_format), \
                    window_end.strftime(file_date_format)))
            if not os.path.exists(csv_path):
                collect_data(\
                    out_path=csv_path, \
                    start_time=window_start.strftime(date_format), \
                    end_time=window_end.strftime(date_format), \
                    show_progress=False)
            X = preprocess(
                csv_path,
                r['window_days'],
                r['predict_range_days'],
                r['lat_granularity'],
                r['lng_granularity'],
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
                    'probability': float(y),
                    'displayName': r['displayName'],
                    'threshold_mag': r['threshold_mag'],
                    'acc': summary['acc'],
                    'auc': summary['auc'],
                    'f1': summary['f1'],
                    'precision': summary['precision'],
                    'recall': summary['recall'],
                })
    
    with open(os.path.join(out_dir, 'predictions.json'), 'w', encoding='utf-8') as o:
        json.dump(out, o, ensure_ascii=False, indent=2)
            
def _midnight(d):
    return datetime(year=d.year, month=d.month, day=d.day, tzinfo=d.tzinfo)

if __name__ == '__main__':
    fire.Fire(main)
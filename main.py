
import os, fire, json
from ml_earthquake import collect_data
from ml_earthquake import preprocess
from ml_earthquake import train
import numpy as np
import random as rn
import tensorflow as tf

def set_random_seed(s):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(s)
    rn.seed(s)
    session_conf = tf.ConfigProto(
        # intra_op_parallelism_threads=1,
        # inter_op_parallelism_threads=1
    )
    from keras import backend as K
    tf.set_random_seed(s)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

def main(
    recipe='recipe.json',
    recipe_id=None,
    data_dir='data',
    out_dir='out',
    work_dir='work',
    log_dir='log',
    random_seed=4126):
    set_random_seed(random_seed)

    if not os.path.exists(recipe):
        raise Exception('recipe: {} does not exists...'.format(recipe))

    data_path = os.path.join(
        data_dir,
        'earthquakes.csv')
    if not os.path.exists(data_path):
        print('collecting earthquake data...')
        collect_data(data_path)

    with open(recipe, 'r', encoding='utf-8') as f:
        recipe_obj = json.load(f)
        for r in recipe_obj['recipe']:
            if recipe_id is not None and recipe_id != r['id']:
                continue
            print('start preprocess and train for recipe ID: {}'.format(r['id']))
            X_train, y_train, X_test, y_test, _, info_test = preprocess(
                data_path,
                r['window_days'],
                r['predict_range_days'],
                r['lat_granularity'],
                r['lng_granularity'],
                r['predict_center_lat'],
                r['predict_center_lng'],
                r['predict_radius_meters'],
                r['threshold_mag'],
                cache_dir=work_dir
            )
            train(
                X_train, y_train,
                X_test, y_test,
                info_test=info_test,
                out_dir=os.path.join(out_dir, r['id']),
                log_dir=log_dir,
                epochs=r['epochs'],
                dropout=r['dropout'],
                random_state=random_seed,
                resampling_method=r['resampling_method'] if 'resampling_method' in r else None
            )

if __name__ == '__main__':
    fire.Fire(main)

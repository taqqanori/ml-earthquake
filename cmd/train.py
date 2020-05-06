import os, fire, json, sys
import numpy as np
import random as rn
import tensorflow as tf

sys.path.append(os.getcwd())
from ml_earthquake import collect_data
from ml_earthquake import preprocess
from ml_earthquake import train

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
    return sess

def main(
    recipe='recipe.json',
    recipe_id=None,
    data_dir='data',
    out_dir='out',
    work_dir='work',
    log_dir='log',
    random_seed=4126):

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
        sess = None
        for r in recipe_obj['recipe']:
            if recipe_id is not None and recipe_id != r['id']:
                continue
            
            if sess is not None:
                sess.close()
                tf.reset_default_graph()
            sess = set_random_seed(random_seed)
            print('start preprocess and train for recipe ID: {}'.format(r['id']))
            X_train, y_train, X_test, y_test, info_train, info_test = preprocess(
                data_path,
                r['window_days'],
                r['predict_range_days'],
                r['lat_granularity'],
                r['lng_granularity'],
                r['predict_center_lat'],
                r['predict_center_lng'],
                r['predict_radius_meters'],
                r['threshold_mag'],
                -90 if r['min_lat'] is None else r['min_lat'],
                90 if r['max_lat'] is None else r['max_lat'],
                -180 if r['min_lng'] is None else r['min_lng'],
                180 if r['max_lng'] is None else r['max_lng'],
                cache_dir=work_dir
            )
            train(
                X_train, y_train,
                X_test, y_test,
                info_train=info_train,
                info_test=info_test,
                out_dir=os.path.join(out_dir, r['id']),
                log_dir=log_dir,
                learning_rate=r['learning_rate'] if 'learning_rate' in r else 5e-6,
                decay=r['decay'] if 'decay' in r else 0.0,
                epochs=r['epochs'],
                dropout=r['dropout'],
                random_state=random_seed,
                resampling_methods=r['resampling_methods'] if 'resampling_methods' in r else None,
                balanced_batch=r['balanced_batch'] if 'balanced_batch' in r else None
            )

if __name__ == '__main__':
    fire.Fire(main)


import os
from ml_earthquake import collect_data
from ml_earthquake import preprocess
from ml_earthquake import train
import numpy as np
import random as rn
import tensorflow as tf

random_seed = 4126

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

if __name__ == '__main__':
    set_random_seed(random_seed)

    data_path = os.path.join(
        'data', 
        'earthquakes.csv')
    if not os.path.exists(data_path):
        print('collecting earthquake data...')
        collect_data(data_path)
    X_train, y_train, X_test, y_test, _, info_test = preprocess(
        data_path,
        15,
        1,
        20,
        30,
        35.680934,
        139.767551,
        150 * 1000,
        4.0,
        cache_dir='work'
    )
    train(
        X_train, y_train, 
        X_test, y_test, 
        info_test=info_test, 
        out_dir='out', 
        log_dir='log',
        random_state=random_seed,
        smote=False,
        under_sampleng=True
    )
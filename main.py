
import os
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
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    from keras import backend as K
    tf.set_random_seed(s)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

if __name__ == '__main__':
    set_random_seed(random_seed)
    X, y, info = preprocess(
        os.path.join(
            'data', 
            'earthquakes.csv'),
        30,
        7,
        100,
        100,
        35.680934,
        139.767551,
        150 * 1000,
        4.0,
        cache_dir=os.path.join(
            'work')
    )
    train(X, y, random_state=random_seed)
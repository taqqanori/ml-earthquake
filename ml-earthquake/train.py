from preprocess import preprocess
import os
import numpy as np
import random as rn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime

def train(X, y, test_size=0.25, epochs=100, log_dir=None, model_path=None, random_state=4126):
    _set_random_seed(random_state)
    index = np.array(range(X.shape[0]))
    X_train, X_test, y_train, y_test, index_train, index_test = \
        train_test_split(X, y, index, test_size=test_size, random_state=random_state)

    model = Sequential()

    model.add(Conv2D(32, 3, input_shape=(X.shape[1], X.shape[1], X.shape[3])))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, 3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, 3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    adam = Adam()
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=["accuracy"])
    model.summary()

    callbacks = [_Reporter(X_test, y_test)]

    if log_dir is not None:
        log_path = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        callbacks.append(TensorBoard(log_dir=log_path))
    if model_path:
        callbacks.append(ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True))

    model.fit(X_train, y_train, \
        epochs=epochs, callbacks=callbacks, \
        validation_data=(X_test, y_test))

def _set_random_seed(s):
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

class _Reporter(Callback):

    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_test)
        tp, fn, fp, tn = confusion_matrix(self.y_test, y_pred >= 0.5).ravel()
        auc = roc_auc_score(self.y_test, y_pred)
        f1 = (2 * tp) / (2 * tp + fp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp /(tp + fp)
        recall = tp / (tp + fn)
        print('AUC: {:.2f}, F1: {:.2f}, acc: {:.2f}, precision: {:.2f}, recall: {:.2f}, TP: {}, FN: {}, FP: {}, TN: {}'.format(\
            auc, f1, acc, precision, recall, tp, fn, fp, tn
        ))

if __name__ == '__main__':
    X, y = preprocess(
        os.path.join(
            os.path.dirname(__file__), 
            '..', 
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
            os.path.dirname(__file__), 
            '..', 
            'work')
    )
    train(X, y)
import os, json
import numpy as np
import random as rn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from keras.callbacks import Callback
from keras.models import Sequential, load_model
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime

date_format = '%Y%m%d'

def train(X, y, info=None, out_dir=None, test_size=0.25, epochs=100, log_dir=None, random_state=4126):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, info_train, info_test = \
        train_test_split(X, y, info, test_size=test_size, random_state=random_state)

    model = Sequential()

    model.add(ConvLSTM2D(
        input_shape=(X.shape[1], X.shape[2], X.shape[3], X.shape[4]),
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        dropout=0.3,
        return_sequences=True
    ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        dropout=0.3,
        return_sequences=True
    ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        dropout=0.3,
        return_sequences=False
    ))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    adam = Adam()
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=["accuracy"])
    model.summary()

    callbacks = [_Reporter(X_test, y_test)]

    model_path = None
    if out_dir is not None:
        model_path = os.path.join(out_dir, 'best_model.h5')
        callbacks.append(
            ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
        )

    if log_dir is not None:
        log_path = os.path.join(log_dir, datetime.now().strftime(date_format + '_%H%M%S'))
        callbacks.append(TensorBoard(log_dir=log_path))

    model.fit(X_train, y_train, \
        epochs=epochs, callbacks=callbacks, \
        validation_data=(X_test, y_test))
    
    if out_dir is not None and info is not None:
        _output(out_dir, X_test, y_test, info_test, model_path)
    
def _output(out_dir, X_test, y_test, info_test, model_path):
    best_model = load_model(model_path)
    acc, auc, f1, precision, recall, tp, fn, fp, tn = _eval(best_model, X_test, y_test)

    # summary
    summary = {
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn),
    }
    _dump(summary, os.path.join(out_dir, 'summary.json'))

    # predictions
    predictions = []
    y_pred = best_model.predict(X_test).reshape(-1)
    for i in range(0, len(y_test)):
        window_start = info_test[i]['window_start'].strftime(date_format)
        window_end = info_test[i]['window_end'].strftime(date_format)
        predict_start = info_test[i]['predict_start'].strftime(date_format)
        predict_end = info_test[i]['predict_end'].strftime(date_format)
        detail_path = 'detail_{}_{}.json'.format(predict_start, predict_end)
        predictions.append({
            'window_start': window_start,
            'window_end': window_end,
            'predict_start': predict_start,
            'predict_end': predict_end,
            'prediction': float(y_pred[i]),
            'fact': float(y_test[i]),
            'detail': detail_path
        })

        # detail
        detail = {
            'mag_heatmaps': [],
            'freq_heatmaps': [],
            'lat_gap': 180 / X_test[i].shape[0],
            'lng_gap': 360 / X_test[i].shape[1],
            'threshold_mag': info_test[i]['threshold_mag'],
            'earthquakes': []
        }
        for win in range(X_test[i].shape[0]):
            mag_heatmap = []
            freq_heatmap = []
            for lat in range(X_test[i].shape[1]):
                for lng in range(X_test[i].shape[2]):
                    mag = X_test[i][win][lat][lng][0]
                    freq = X_test[i][win][lat][lng][1]
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
            detail['mag_heatmaps'].append(mag_heatmap)
            detail['freq_heatmaps'].append(freq_heatmap)
        for eq in info_test[i]['earthquakes']:
            detail['earthquakes'].append({
                'time': eq['time'].strftime(date_format),
                'latitude': eq['latitude'],
                'longitude': eq['longitude'],
                'mag': eq['mag']
            })
        _dump(detail, os.path.join(out_dir, detail_path))
    _dump(predictions, os.path.join(out_dir, 'predictions.json'))

def _dump(o, path):
    with open(path, 'w') as f:
        json.dump(o, f, indent=2)

class _Reporter(Callback):

    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        acc, auc, f1, precision, recall, tp, fn, fp, tn = _eval(self.model, self.X_test, self.y_test)
        print('\nAUC: {:.2f}, F1: {:.2f}, acc: {:.2f}, precision: {:.2f}, recall: {:.2f}, TP: {}, FN: {}, FP: {}, TN: {}'.format(\
            auc, f1, acc, precision, recall, tp, fn, fp, tn
        ))

def _eval(model, X_test, y_test):
    y_pred = model.predict(X_test).reshape(-1)
    tp, fn, fp, tn = confusion_matrix(y_test, y_pred >= 0.5).ravel()
    auc = roc_auc_score(y_test, y_pred)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp /(tp + fp)
    recall = tp / (tp + fn)
    return acc, auc, f1, precision, recall, tp, fn, fp, tn
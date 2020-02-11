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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import (
    RandomUnderSampler,
    ClusterCentroids,
    NearMiss,
    TomekLinks,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    AllKNN,
    NeighbourhoodCleaningRule,
)
from imblearn.keras import BalancedBatchGenerator

date_format = '%Y%m%d'

def train(
    X_train,
    y_train,
    X_test,
    y_test,
    info_train=None,
    info_test=None,
    out_dir=None,
    model_file_name='best_model.h5',
    learning_rate=5e-6,
    decay=0.0,
    epochs=100,
    dropout=0.3,
    log_dir=None,
    resampling_methods=None,
    balanced_batch=None,
    use_class_weight=False,
    random_state=4126):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    model = Sequential()

    model.add(ConvLSTM2D(
        input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]),
        filters=30,
        kernel_size=(3, 3),
        padding='same',
        dropout=dropout,
        return_sequences=False
    ))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(lr=learning_rate, decay=decay)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=["accuracy"])
    model.summary()

    callbacks = [_Reporter(X_test, y_test)]

    model_path = None
    if out_dir is not None:
        model_path = os.path.join(out_dir, model_file_name)
        callbacks.append(
            ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
        )

    if log_dir is not None:
        log_path = os.path.join(log_dir, datetime.now().strftime(date_format + '_%H%M%S'))
        callbacks.append(TensorBoard(log_dir=log_path))
    
    positive = (0.5 <= y_train).sum()
    negative = (y_train < 0.5).sum()
    print('train data balance P:{} : N:{}'.format(positive, negative))
    class_weight = {
        0: positive / (positive + negative),
        1: negative / (positive + negative),
    } if use_class_weight else None

    if resampling_methods is not None:
        for resampling_method in resampling_methods:
            X_train, y_train = _resample(X_train, y_train, resampling_method, random_state)

    if balanced_batch == True:
        print('using BalancedBatchGenerator')
        batch_gen = BalancedBatchGenerator(\
            X_train, y_train, \
            sampler=RandomUnderSamplerWrapper(), random_state=random_state)
        model.fit_generator(generator=batch_gen, \
            epochs=epochs, callbacks=callbacks, \
            validation_data=(X_test, y_test),
            class_weight=class_weight)
    else:
        model.fit(X_train, y_train, \
            epochs=epochs, callbacks=callbacks, \
            validation_data=(X_test, y_test),
            class_weight=class_weight)
    
    if out_dir is not None and info_train is not None and info_test is not None:
        _output(out_dir, X_test, y_test, info_train, info_test, model_path)
    
def _output(out_dir, X_test, y_test, info_train, info_test, model_path):
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
        'train_start_date': info_train[0]['window_start'].strftime(date_format),
        'train_end_date': info_train[-1]['predict_end'].strftime(date_format),
        'test_start_date': info_test[0]['window_start'].strftime(date_format),
        'test_end_date': info_test[-1]['predict_end'].strftime(date_format),
    }
    _dump(summary, os.path.join(out_dir, 'summary.json'))

    # validations
    validations = []
    y_pred = best_model.predict(X_test).reshape(-1)
    for i in range(0, len(y_test)):
        window_start = info_test[i]['window_start'].strftime(date_format)
        window_end = info_test[i]['window_end'].strftime(date_format)
        predict_start = info_test[i]['predict_start'].strftime(date_format)
        predict_end = info_test[i]['predict_end'].strftime(date_format)
        detail_path = 'detail_{}_{}.json'.format(predict_start, predict_end)
        validation = {
            'window_start': window_start,
            'window_end': window_end,
            'predict_start': predict_start,
            'predict_end': predict_end,
            'prediction': float(y_pred[i]),
            'fact': float(y_test[i]),
            'detail': detail_path
        }
        validations.append(validation)

        # detail
        detail = {
            'window_start': window_start,
            'window_end': window_end,
            'predict_start': predict_start,
            'predict_end': predict_end,
            'mag_heatmaps': [],
            'freq_heatmaps': [],
            'depth_heatmaps': [],
            'lat_gap': 180 / X_test[i].shape[1],
            'lng_gap': 360 / X_test[i].shape[2],
            'threshold_mag': info_test[i]['threshold_mag'],
            'earthquakes': []
        }
        for win in range(X_test[i].shape[0]):
            mag_heatmap = []
            freq_heatmap = []
            depth_heatmap = []
            for lat in range(X_test[i].shape[1]):
                for lng in range(X_test[i].shape[2]):
                    mag = X_test[i][win][lat][lng][0]
                    freq = X_test[i][win][lat][lng][1]
                    depth = X_test[i][win][lat][lng][2]
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
        max_mag = 0
        for eq in info_test[i]['earthquakes']:
            detail['earthquakes'].append({
                'time': eq['time'].strftime(date_format + '%H%M%S'),
                'latitude': eq['latitude'],
                'longitude': eq['longitude'],
                'depth': eq['depth'],
                'mag': eq['mag']
            })
            if max_mag < eq['mag']:
                max_mag = eq['mag']
        validation['max_mag'] = max_mag
        _dump(detail, os.path.join(out_dir, detail_path))
    _dump({ 'validations': validations }, os.path.join(out_dir, 'validations.json'))

def _resample(X_train, y_train, resampling_method, random_state):
    resampler = _get_resampler(resampling_method, random_state)
    print('performing {}...'.format(resampling_method['name']))
    X_train_resample, y_train = resampler.fit_sample(X_train.reshape(X_train.shape[0], -1), y_train)
    X_train = X_train_resample.reshape(\
        X_train_resample.shape[0], \
        X_train.shape[1], \
        X_train.shape[2], \
        X_train.shape[3], \
        X_train.shape[4])
    positive = (0.5 <= y_train).sum()
    negative = (y_train < 0.5).sum()
    print('{} performed train data balance P:{} : N:{}'.format(
        resampling_method['name'], positive, negative))
    return X_train, y_train

def _get_resampler(method, random_state, n_jobs=8):
    args = {
        'random_state': random_state,
        'n_jobs': n_jobs
    }
    if 'args' in method:
        args = {
            **args,
            **method['args']
        }
    if method['name'] == 'SMOTE':
        return SMOTE(**args)
    elif method['name'] == 'RandomUnderSampler':
        del args['n_jobs']
        return RandomUnderSampler(**args) # pylint: disable=unexpected-keyword-arg
    elif method['name'] == 'ClusterCentroids':
        return ClusterCentroids(**args)
    elif method['name'] == 'NearMiss':
        return NearMiss(**args)
    elif method['name'] == 'TomekLinks':
        return TomekLinks(**args)
    elif method['name'] == 'CondensedNearestNeighbour':
        return CondensedNearestNeighbour(**args)
    elif method['name'] == 'EditedNearestNeighbours':
        return EditedNearestNeighbours(**args)
    elif method['name'] == 'AllKNN':
        return AllKNN(**args)
    elif method['name'] == 'NeighbourhoodCleaningRule':
        return NeighbourhoodCleaningRule(**args)
    else:
        raise Exception('unknown resampler: {}'.format(method['name']))

class RandomUnderSamplerWrapper(RandomUnderSampler):

    def fit_resample(self, X, y):
        X_resample, y = super().fit_sample(X.reshape(X.shape[0], -1), y)
        X = X_resample.reshape(\
            X_resample.shape[0], \
            X.shape[1], \
            X.shape[2], \
            X.shape[3], \
            X.shape[4])
        return X, y

def _dump(o, path):
    with open(path, 'w') as f:
        json.dump(o, f, indent=2)

class _Reporter(Callback):

    def __init__(self, X_test, y_test, monitor='val_loss', best_only=True):
        self.X_test = X_test
        self.y_test = y_test
        self.monitor = monitor
        self.best_only = best_only
        self.best = np.inf

    def on_epoch_end(self, epoch, logs={}):
        if self.best_only:
            score = logs.get(self.monitor)
            if self.best < score:
                return
            self.best = score
        acc, auc, f1, precision, recall, tp, fn, fp, tn = _eval(self.model, self.X_test, self.y_test)
        print('Epoch {}: AUC: {:.3f}, F1: {:.3f}, acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, TP: {}, FN: {}, FP: {}, TN: {}'.format(\
            epoch + 1, auc, f1, acc, precision, recall, tp, fn, fp, tn
        ))

def _eval(model, X_test, y_test):
    y_pred = model.predict(X_test).reshape(-1)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred >= 0.5).ravel()
    auc = roc_auc_score(y_test, y_pred)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp /(tp + fp)
    recall = tp / (tp + fn)
    return acc, auc, f1, precision, recall, tp, fn, fp, tn

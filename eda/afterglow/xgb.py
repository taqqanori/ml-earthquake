import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

def xgb(col):
    y_labels = [
        'Tokyo 200km M4',
        'Tokyo 200km M5',
        'Tokyo 500km M4',
        'Tokyo 500km M5',
        'Tokyo 1000km M4',
        'Tokyo 1000km M5'
    ]
    df = pd.read_csv('data/afterglow_table.csv')
    X = df
    for y_label in y_labels:
        X = X.drop(y_label, axis=1)
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, train_size=0.75)
    model = XGBClassifier()
    model = model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:,1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred >= 0.5).ravel()
    auc = roc_auc_score(y_test, y_pred)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp /(tp + fp)
    recall = tp / (tp + fn)

    print('AUC: {:.3f}, F1: {:.3f}, acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, TP: {}, FN: {}, FP: {}, TN: {}'.format(\
        auc, f1, acc, precision, recall, tp, fn, fp, tn
    ))

    _, ax = plt.subplots(figsize=(12, 4))
    plot_importance(model, ax=ax, importance_type='gain', show_values=False, max_num_features=50)
    plt.show()

if __name__ == '__main__':
    xgb('Tokyo 500km M4')

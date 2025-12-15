import numpy as np
import random
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class Stacking(BaseEstimator, ClassifierMixin):
    def __init__(self, models, n_folds, meta_model):
        self.models = models
        self.n_folds = n_folds
        self.meta_model = meta_model

    def _shuffle(self, array):
        n = len(array)
        for i in range(n - 1, 0, -1):
            j = random.randint(0, i)
            array[i], array[j] = array[j], array[i]

    def _create_folds(self, n):
        idx = np.arange(n)
        self._shuffle(idx)
        q = n // self.n_folds
        r = n % self.n_folds
        folds_idx = []
        start = 0
        for fold_idx in range(self.n_folds):
            fold_size = q + 1 if fold_idx < r else q
            end = fold_size + start
            val_idx = idx[start : end]
            train_idx = np.concatenate([idx[:start], idx[end:]])
            folds_idx.append((train_idx, val_idx))
            start = end
        return folds_idx

    def fit(self, X, y):
        n = X.shape[0]
        l = len(self.models)
        self.c = len(np.unique(y))
        predictions = np.zeros(shape=(n, l * self.c))
        folds = self._create_folds(n)
        for i, model in enumerate(self.models):
            model_predictions = np.zeros(shape=(n, self.c))
            for train_idx, val_idx in folds:
                X_train = X[train_idx]
                X_val = X[val_idx] 
                y_train = y[train_idx]
                cloned_model = clone(model)
                cloned_model.fit(X_train, y_train)
                y_pred = cloned_model.predict_proba(X_val)
                model_predictions[val_idx] = y_pred
            predictions[:, i * self.c:(i + 1) * self.c] = model_predictions
            model.fit(X, y)
        
        self.meta_model.fit(predictions, y)
        return self


    def predict(self, X):
        n = X.shape[0]
        l = len(self.models)
        models_predictions = np.zeros(shape=(n, l * self.c))
        for i, model in enumerate(self.models):
            models_predictions[:, i * self.c:(i + 1) * self.c] = model.predict_proba(X)
        prediction = self.meta_model.predict(models_predictions)
        return prediction

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

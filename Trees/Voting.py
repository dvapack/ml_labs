import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class Voting(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights, mode):
        self.models = models
        self.weights = weights
        self.mode = mode

    def fit(self, X, y):
        self.classes = np.unique(y)
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        n = X.shape[0]
        n_models = len(self.models)
        weights = self.weights
        if weights is None:
            weights = np.ones(n_models)
        else:
            weights = np.asarray(weights)
        if self.mode == "hard":
            hard_predictions = np.zeros((n, n_models))
            for i, model in enumerate(self.models):
                hard_predictions[:, i] = model.predict(X)
            classes = self.classes
            one_hot = (hard_predictions[:, :, None] == classes[None, None, :])
            weighted = one_hot * weights[None, :, None]
            V = weighted.sum(axis=1)
            best_class_indices = np.argmax(V, axis=1)
            y_pred = classes[best_class_indices]
        else:
            soft_predictions = np.zeros((n, n_models, len(self.classes)))
            for i, model in enumerate(self.models):
                soft_predictions[:, i, :] = model.predict_proba(X)
            weighted = soft_predictions * weights[None, :, None]
            V = weighted.sum(axis=1)
            y_pred = np.argmax(V, axis=1)
            y_pred = self.classes[y_pred]
        return y_pred
    
    def predict_proba(self, X):
        n = X.shape[0]
        n_models = len(self.models)
        weights = self.weights
        if weights is None:
            weights = np.ones(n_models)
        else:
            weights = np.asarray(weights)
        
        soft_predictions = np.zeros((n, n_models, len(self.classes)))
        for i, model in enumerate(self.models):
            soft_predictions[:, i, :] = model.predict_proba(X)
        weighted = soft_predictions * weights[None, :, None]
        V = weighted.sum(axis=1)
        return V / V.sum(axis=1, keepdims=True)
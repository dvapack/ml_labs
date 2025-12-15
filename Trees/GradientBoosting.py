import numpy as np
from Losses import Loss
from Trees import Tree
from sklearn.base import BaseEstimator, ClassifierMixin

class MyGradientBoosting(BaseEstimator, ClassifierMixin):
    def __init__(self, loss=None, learning_rate=0.1, n_estimators=100, subsampling=1.0, max_depth=3, min_samples_leaf=1, min_gain_to_split=0.0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.base_prediction = None
        self.trees = []
        self.n_estimators = n_estimators
        self.subsampling = subsampling
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_to_split = min_gain_to_split
        self.tree_params = {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_gain_to_split": min_gain_to_split
        }

    def _random_subset(self, n):
        size = int(self.subsampling * n)
        sample = np.random.choice(n, size, replace=False)
        return sample

    def fit(self, X, y):
        if self.loss is None:
            from Losses import MSELoss
            self.loss = MSELoss()
        
        n = X.shape[0]
        self.base_prediction = self.loss.base_predictions(y)
        F = np.full(shape=n, fill_value=self.base_prediction)

        for t in range(self.n_estimators):
            subset_idx = self._random_subset(n)
            residuals = np.zeros(n, dtype=float)
            for i in subset_idx:
                residuals[i] = -self.loss.gradients(F[i], y[i])

            tree = Tree(
                max_depth = self.tree_params["max_depth"],
                min_samples_leaf = self.tree_params["min_samples_leaf"],
                min_gain_to_split = self.tree_params["min_gain_to_split"]
            )
            tree.fit(X[subset_idx], residuals[subset_idx])

            tree_preds = tree.predict(X)
            F = F + self.learning_rate * tree_preds

            self.trees.append(tree)
        
        return self

    def predict(self, X):
        n = X.shape[0]
        predictions = np.full(shape=n, fill_value=self.base_prediction)
        for tree in self.trees:
            tree_predict = tree.predict(X)
            predictions = predictions + self.learning_rate * tree_predict
        
        return predictions
    
    def predict_proba(self, X):
        pred = self.predict(X)
        n_samples = pred.shape[0]
        n_classes = 2
        proba = np.zeros((n_samples, n_classes))
        proba[:, 1] = 1 / (1 + np.exp(-pred))
        proba[:, 0] = 1 - proba[:, 1]
        return proba


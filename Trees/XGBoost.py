import numpy as np
from Losses import Loss
from Trees import XGBoostTree
from sklearn.base import BaseEstimator, ClassifierMixin

class MyXGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, loss=None, learning_rate=0.1, n_estimators=100, subsampling=1.0, missing_value=None, gamma=0, reg_lambda=1, reg_alpha=0, max_depth=3, min_child_weight=1):
        self.loss = loss
        self.learning_rate = learning_rate
        self.base_prediction = None
        self.trees = []
        self.n_estimators = n_estimators
        self.subsampling = subsampling
        self.missing_value = missing_value
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.tree_params = {
            "missing_value": missing_value,
            "gamma": gamma,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight
        }

    def _random_subset(self, n):
        size = int(self.subsampling * n)
        sample = np.random.choice(n, size, replace=False)
        return sample

    def fit(self, X, y):
        if self.loss is None:
            from Losses import LogisticLoss
            self.loss = LogisticLoss()
        
        n = X.shape[0]
        self.base_prediction = self.loss.base_predictions(y)
        F = np.full(shape=n, fill_value=self.base_prediction)

        for t in range(self.n_estimators):
            subset_idx = self._random_subset(n)
            g = np.zeros(n, dtype=float)
            h = np.zeros(n, dtype=float)
            for i in subset_idx:
                g[i] = self.loss.gradients(F[i], y[i])
                h[i] = self.loss.hessians(F[i], y[i])

            tree = XGBoostTree(
                missing_value = self.tree_params["missing_value"],
                gamma = self.tree_params["gamma"],
                reg_lambda = self.tree_params["reg_lambda"],
                reg_alpha = self.tree_params["reg_alpha"],
                max_depth = self.tree_params["max_depth"],
                min_child_weight = self.tree_params["min_child_weight"],
            )
            tree.fit(X[subset_idx], g[subset_idx], h[subset_idx])

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
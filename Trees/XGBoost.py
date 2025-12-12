import numpy as np
from Losses import Loss
from Trees.Trees import XGBoostTree

class MyXGBoost:
    def __init__(self, loss: Loss, learning_rate, n_estimators, subsampling, **tree_params):
        self.loss = loss
        self.learning_rate = learning_rate
        self.base_prediction = None
        self.trees = []
        self.n_estimators = n_estimators
        self.subsampling = subsampling
        self.tree_params = tree_params

    def _random_subset(self, n):
        size = int(self.subsampling * n)
        sample = np.random.choice(n, size, replace=False)
        return sample

    def fit(self, X, y):
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
                min_gain_to_split = self.tree_params["min_gain_to_split"]
            )
            tree.fit(X[subset_idx], g[subset_idx], h[subset_idx])

            tree_preds = tree.predict(X)
            F = F + self.learning_rate * tree_preds

            self.trees.append(tree)

    def predict(self, X):
        n = X.shape[0]
        predictions = np.full(shape=n, fill_value=self.base_prediction)
        for tree in self.trees:
            tree_predict = tree.predict(X)
            predictions = predictions + self.learning_rate * tree_predict
        
        return predictions
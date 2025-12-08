import numpy as np
from Losses import Loss
from Tree import Tree

class MyGradientBoosting:
    def __init__(self, loss: Loss, learning_rate, n_estimators, **tree_params):
        self.loss = loss
        self.learning_rate = learning_rate
        self.base_prediction = None
        self.trees = []
        self.n_estimators = n_estimators
        self.tree_params = tree_params

    def fit(self, X, y):
        n = X.shape[0]
        self.base_prediction = self.loss.base_predictions(y)
        F = np.full(shape=n, fill_value=self.base_prediction)

        for t in range(self.n_estimators):
            residuals = np.zeros(n, dtype=float)
            for i in range(n):
                residuals[i] = -self.loss.gradients(F[i], y[i])

            tree = Tree(
                max_depth = self.tree_params["max_depth"],
                min_samples_leaf = self.tree_params["min_samples_leaf"],
                min_gain_to_split = self.tree_params["min_gain_to_split"]
            )
            tree.fit(X, residuals)

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


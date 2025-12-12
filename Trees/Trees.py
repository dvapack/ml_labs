import numpy as np

class Node:
    def __init__(self, is_leaf, value, feature, threshold, left: None, right: None):
        self.is_leaf = is_leaf
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def predict(self, X):
        if self.is_leaf:
            return self.value
        else:
            if X[self.feature] <= self.threshold:
                return self.left.predict(X)
            else:
                return self.right.predict(X)
            


class Tree:
    def __init__(self, max_depth, min_samples_leaf, min_gain_to_split):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_to_split = min_gain_to_split

    def _build_node(self, X, r, S, depth, n_features):
        n_S = len(S)
        if n_S < 2 * self.min_samples_leaf or depth >= self.max_depth:
            nu = np.mean(r[S]) if n_S > 0 else 0
            return Node(is_leaf=True, value=nu, feature=None, threshold=None, left=None, right=None)
        
        best_gain = - np.inf
        best_feature = None
        best_threshold = None
        best_L = None
        best_R = None
        for j in range(n_features):
            pairs = [(X[i][j], r[i], i) for i in S]
            pairs.sort()
            sum_left = 0
            count_left = 0
            sum_right = np.sum(r[S])
            count_right = n_S

            for k in range(n_S - 1):
                (_, r_val, idx) = pairs[k]
                sum_left += r_val
                count_left += 1
                sum_right -= r_val
                count_right -= 1

                if count_left < self.min_samples_leaf or count_right < self.min_samples_leaf:
                    continue

                total_sum = sum_left + sum_right
                gain = (sum_left**2 / count_left) + (sum_right**2 / count_right) - (total_sum**2 / n_S)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    next_val = pairs[k+1][0]
                    best_threshold = (pairs[k][0] + next_val) / 2
                    
                    best_L = [i[2] for i in pairs[0:k+1]]
                    best_R = [i[2] for i in pairs[k+1:]]

        if best_gain < self.min_gain_to_split:
            nu = (1 / n_S) * sum(r)
            return Node(is_leaf=True, value=nu, feature=None, threshold=None, left=None, right=None)
        
        left_child = self._build_node(X, r, best_L, depth+1, n_features)
        right_child = self._build_node(X, r, best_R, depth+1, n_features)

        return Node(
            value=None,
            is_leaf=False,
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def fit(self, X, r):
        n = X.shape[0]
        d = X.shape[1]
        root_idx = np.arange(0, n)
        self.root = self._build_node(X=X, r=r, S=root_idx, depth=0, n_features=d)

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.root.predict(X[i]))
        return np.array(predictions, dtype=float)
        

class XGBoostTree:
    def __init__(self, reg_lambda, reg_alpha, gamma, max_depth, min_child_weight, min_gain_to_split):
        self.root = None
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.min_gain_to_split = min_gain_to_split

    def _compute_weight(self, G, H):
        if G > self.reg_alpha:
            return - (G - self.reg_alpha) / (H + self.reg_lambda)
        elif G < -self.reg_alpha:
            return - (G + self.reg_alpha) / (H + self.reg_lambda)
        else:
            return 0

    def _build_node(self, X, g, h, S, depth, n_features):
        G = np.sum(g[S])
        H = np.sum(h[S])
        n_S = len(S)

        if depth >= self.max_depth:
            w = self._compute_weight(G, H)
            return Node(is_leaf=True, value=w, feature=None, 
                        threshold=None, left=None, right=None)

        best_gain = - np.inf
        best_feature = None
        best_threshold = None
        best_L = None
        best_R = None
        for j in range(n_features):
            pairs = [(X[i][j], g[i], h[i], i) for i in S]
            pairs.sort()

            G_L = 0
            H_L = 0
            G_R = G
            H_R = H

            for k in range(n_S - 1):
                (_, g_val, h_val, idx) = pairs[k]
                G_L += g_val
                H_L += h_val
                G_R -= g_val
                H_R -= h_val

                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue

                gain = 0.5 * (
                    G_L * G_L / (H_L + self.reg_lambda) +
                    G_R * G_R / (H_R + self.reg_lambda) -
                    G * G / (H + self.reg_lambda)
                ) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    next_val = pairs[k+1][0]
                    best_threshold = (pairs[k][0] + next_val) / 2
                    
                    best_L = [i[3] for i in pairs[0:k+1]]
                    best_R = [i[3] for i in pairs[k+1:]]

        if best_gain < self.min_gain_to_split:
            w = self._compute_weight(G, H)
            return Node(is_leaf=True, value=w, feature=None, 
                        threshold=None, left=None, right=None)
        

        left_child = self._build_node(X, g, h, best_L, depth+1, n_features)
        right_child = self._build_node(X, g, h, best_R, depth+1, n_features)

        return Node(
            value=None,
            is_leaf=False,
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def fit(self, X, g, h):
        n = X.shape[0]
        d = X.shape[1]
        root_idx = np.arange(0, n)
        self.root = self._build_node(X, g, h, root_idx, depth=0, n_features=d)

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.root.predict(X[i]))
        return np.array(predictions, dtype=float)

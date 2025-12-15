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

class XGBoostNode:
    def __init__(self, is_leaf, value, feature, threshold, missing_value, default_left, left: None, right: None):
        self.is_leaf = is_leaf
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.missing_value = missing_value
        self.default_left = default_left

    def predict(self, X):
        if self.is_leaf:
            return self.value
        val = X[self.feature]
        is_missing = (
            (self.missing_value is np.nan and np.isnan(val)) or
            (val == self.missing_value)
        )
        if is_missing:
            child = self.left if self.default_left else self.right
        else:
            child = self.left if val < self.threshold else self.right
        return child.predict(X)
            


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
    def __init__(self, reg_lambda, reg_alpha, gamma, missing_value, max_depth, min_child_weight):
        self.root = None
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.missing_value = missing_value
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight

    def _compute_weight(self, G, H):
        if H + self.reg_lambda == 0:
            return 0.0
        if G > self.reg_alpha:
            return - (G - self.reg_alpha) / (H + self.reg_lambda)
        elif G < -self.reg_alpha:
            return - (G + self.reg_alpha) / (H + self.reg_lambda)
        else:
            return 0

    def _build_node(self, X, g, h, S, depth, n_features, missing_value):
        G = np.sum(g[S])
        H = np.sum(h[S])
        n_S = len(S)

        if depth >= self.max_depth:
            w = self._compute_weight(G, H)
            return XGBoostNode(is_leaf=True, value=w, feature=None, 
                        threshold=None, left=None, right=None, missing_value=None, default_left=None)

        best_gain = - np.inf
        best_feature = None
        best_threshold = None
        best_default_left = None
        best_L = None
        best_R = None
        for j in range(n_features):
            known = []
            missing = []
            for i in S:
                if X[i][j] == missing_value or np.isnan(X[i][j]):
                    missing.append((g[i], h[i], i))
                else:
                    known.append((X[i][j], g[i], h[i], i))
            known.sort(key=lambda x: x[0])
            # pairs = [(X[i][j], g[i], h[i], i) for i in S]
            # pairs.sort()
            G_known = sum(item[1] for item in known)
            H_known = sum(item[2] for item in known)
            G_L = 0
            H_L = 0
            G_R = G_known
            H_R = H_known
            n_known = len(known)
            for k in range(n_known - 1):
                (x_val, g_val, h_val, idx) = known[k]
                G_L += g_val
                H_L += h_val
                G_R -= g_val
                H_R -= h_val

                if x_val == known[k + 1][0]:
                    continue

                G_miss = sum(item[0] for item in missing)
                H_miss = sum(item[1] for item in missing)

                G_L1 = G_L + G_miss
                H_L1 = H_L + H_miss
                G_R1 = G_R
                H_R1 = H_R

                if H_L1 >= self.min_child_weight and H_R1 >= self.min_child_weight:
                    gain1 = 0.5 * (
                        G_L1 * G_L1 / (H_L1 + self.reg_lambda) +
                        G_R1 * G_R1 / (H_R1 + self.reg_lambda) -
                        G * G / (H + self.reg_lambda)
                    ) - self.gamma
                else:
                    gain1 = - np.inf
                
                G_L2 = G_L
                H_L2 = H_L
                G_R2 = G_R + G_miss
                H_R2 = H_R + H_miss

                if H_L2 >= self.min_child_weight and H_R2 >= self.min_child_weight:
                    gain2 = 0.5 * (
                        G_L2 * G_L2 / (H_L2 + self.reg_lambda) +
                        G_R2 * G_R2 / (H_R2 + self.reg_lambda) -
                        G * G / (H + self.reg_lambda)
                    ) - self.gamma
                else:
                    gain2 = - np.inf

                if gain1 > gain2:
                    current_gain = gain1
                    default_left = True
                else:
                    current_gain = gain2
                    default_left = False

                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature = j
                    next_val = known[k+1][0]
                    best_threshold = (known[k][0] + next_val) / 2
                    best_default_left = default_left

                    known_L = [i[3] for i in known[0:k+1]]
                    known_R = [i[3] for i in known[k+1:]]
                    
                    if default_left:
                        best_L = known_L + [i[2] for i in missing]
                        best_R = known_R
                    else:
                        best_L = known_L
                        best_R = known_R + [i[2] for i in missing]                       

        if best_gain <= 0 or best_feature is None:
            w = self._compute_weight(G, H)
            return XGBoostNode(is_leaf=True, value=w, feature=None, 
                        threshold=None, left=None, right=None, missing_value=None, default_left=None)
        

        left_child = self._build_node(X, g, h, best_L, depth+1, n_features, missing_value)
        right_child = self._build_node(X, g, h, best_R, depth+1, n_features, missing_value)

        return XGBoostNode(
            value=None,
            is_leaf=False,
            feature=best_feature,
            threshold=best_threshold,
            missing_value=missing_value,
            default_left=best_default_left,
            left=left_child,
            right=right_child
        )

    def fit(self, X, g, h):
        n = X.shape[0]
        d = X.shape[1]
        root_idx = np.arange(0, n)
        self.root = self._build_node(X, g, h, root_idx, depth=0, n_features=d, missing_value=self.missing_value)

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.root.predict(X[i]))
        return np.array(predictions, dtype=float)

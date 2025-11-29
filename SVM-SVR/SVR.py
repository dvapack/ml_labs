import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from kernels import linear_kernel
from sklearn.metrics import mean_squared_error

class MySVR(BaseEstimator, RegressorMixin):
    """
    Класс регрессора с помощью Support Vector Machine.

    Для решения двойственной задачи используется модифицированный Simplified SMO.

    The Simplified SMO Algorithm http://cs229.stanford.edu/materials/smo.pdf
    """
    def __init__(self, C=1.0, epsilon=0.1, kernel=linear_kernel, tol=1e-3, max_iter=1000, **kernel_args):
        """
        SVR с использованием модифицированного Simplified SMO и поддержкой sklearn.GridSearchCV

        Args:
            C (float): Параметр регуляризации. Default = 1.0.
            epsilon (float): Ширина трубки для предсказаний. Default = 0.1.
            kernel (sklearn or self-made): Ядро. Default = linear_kernel.
            tol (float): Точность сходимости. Default = 1e-3.
            max_iter (int): Число итераций. Default = 1000
            **kernel_args: Гиперпараметры ядра.
        Returns:
            Объект регрессора.
        """
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter
        self.kernel_args = kernel_args

        self.b = 0
        self.alpha = None
        self.alpha_star = None
        self.n_samples = 0
        self.support_vectors = None
        self.support_vector_alphas = None
        self.support_vector_alpha_stars = None

    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray):
        """Метод вызова ядра"""
        if callable(self.kernel):
            return self.kernel(x1, x2, **self.kernel_args)
        else:
            raise ValueError(f"Не поддерживаемое ядро: {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Метод для "обучения" регрессор.

        Args:
            X (np.ndarray): Массив фичей.
            y (np.ndarray): Массив меток классов.
        Returns:
            "Обученный" регрессор.
        """
        X, y = check_X_y(X, y)
        self.n_samples = X.shape[0]

        self.K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                self.K[i, j] = self._kernel_function(X[i], X[j])

        self.alpha = np.zeros(self.n_samples)
        self.alpha_star = np.zeros(self.n_samples)
        self.b = 0

        iter_count = 0
        while iter_count < self.max_iter:
            num_changed = 0
            for i in range(self.n_samples):
                f_i = np.sum((self.alpha - self.alpha_star) * self.K[:, i]) + self.b
                E_i = f_i - y[i]

                # alpha_i: если E_i > epsilon, то alpha_i < C
                # alpha_i*: если E_i < -epsilon, то alpha_i* < C
                if (E_i > self.epsilon and self.alpha[i] < self.C) or \
                   (E_i < -self.epsilon and self.alpha_star[i] < self.C) or \
                   (-self.epsilon <= E_i <= self.epsilon and (self.alpha[i] > 0 or self.alpha_star[i] > 0)):
                    
                    j = i
                    while j == i:
                        j = np.random.randint(0, self.n_samples)

                    f_j = np.sum((self.alpha - self.alpha_star) * self.K[:, j]) + self.b
                    E_j = f_j - y[j]

                    alpha_i_old = self.alpha[i]
                    alpha_i_star_old = self.alpha_star[i]
                    alpha_j_old = self.alpha[j]
                    alpha_j_star_old = self.alpha_star[j]

                    # L H
                    if y[i] == y[j]:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)
                    else:
                        L = max(0, alpha_j_old - alpha_i_star_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_star_old)

                    if L == H:
                        continue

                    # eta
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue

                    diff_i = alpha_i_old - alpha_i_star_old
                    diff_j = alpha_j_old - alpha_j_star_old

                    diff_j_new = diff_j - (E_i - E_j) / eta

                    total = diff_i + diff_j
                    diff_j_clipped = np.clip(diff_j_new, -self.C, self.C)
                    diff_i_new = total - diff_j_clipped
                    # либо alpha, либо alpha* = 0
                    if diff_i_new >= 0:
                        self.alpha[i] = diff_i_new
                        self.alpha_star[i] = 0
                    else:
                        self.alpha[i] = 0
                        self.alpha_star[i] = -diff_i_new

                    if diff_j_clipped >= 0:
                        self.alpha[j] = diff_j_clipped
                        self.alpha_star[j] = 0
                    else:
                        self.alpha[j] = 0
                        self.alpha_star[j] = -diff_j_clipped

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5 and abs(self.alpha_star[j] - alpha_j_star_old) < 1e-5:
                        continue

                    # b
                    b1 = self.b - E_i - (self.alpha[i] - alpha_i_old) * self.K[i, i] \
                         - (self.alpha[j] - alpha_j_old) * self.K[i, j] \
                         + (self.alpha_star[i] - alpha_i_star_old) * self.K[i, i] \
                         + (self.alpha_star[j] - alpha_j_star_old) * self.K[i, j]

                    b2 = self.b - E_j - (self.alpha[i] - alpha_i_old) * self.K[i, j] \
                         - (self.alpha[j] - alpha_j_old) * self.K[j, j] \
                         + (self.alpha_star[i] - alpha_i_star_old) * self.K[i, j] \
                         + (self.alpha_star[j] - alpha_j_star_old) * self.K[j, j]

                    if 0 < self.alpha[i] < self.C or 0 < self.alpha_star[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C or 0 < self.alpha_star[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed += 1

            iter_count += 1
            if num_changed == 0:
                break

        sv_mask = (self.alpha > 1e-5) | (self.alpha_star > 1e-5)
        self.support_vectors = X[sv_mask]
        self.support_vector_alphas = self.alpha[sv_mask]
        self.support_vector_alpha_stars = self.alpha_star[sv_mask]

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Метод, который делает предсказания для массива фичей.

        Args:
            X (np.ndarray): Массив фичей.

        Returns:
            np.ndarray
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        kernel_matrix = np.zeros((X.shape[0], len(self.support_vectors)))
        for i in range(X.shape[0]):
            for j, sv in enumerate(self.support_vectors):
                kernel_matrix[i, j] = self._kernel_function(X[i], sv)

        predictions = np.dot(kernel_matrix, self.support_vector_alphas - self.support_vector_alpha_stars) + self.b
        return predictions

    def score(self, X: np.ndarray, y: np.ndarray):
        """
        Какой-то метод для нахождения метрики. Видимо для выбора лучших гиперпараметров
        Args:
            X (np.ndarray): Массив фичей.
            y (np.ndarray): Массив меток класса.

        Returns:
            score (float):
            (Sklearn) The R^2 score or ndarray of scores if 'multioutput' is 'raw_values'.
        """
        return mean_squared_error(y, self.predict(X))

    def get_params(self, deep=True):
        """
        Метод для sklearn.GridSearchCV
        """
        params = {
            'C': self.C,
            'epsilon': self.epsilon,
            'kernel': self.kernel,
            'tol': self.tol,
            'max_iter': self.max_iter
        }
        params.update(self.kernel_args)
        return params

    def set_params(self, **params):
        """
        Метод для sklearn.GridSearchCV
        """
        kernel_args_params = {}
        regular_params = {}
        valid_params = ['C', 'epsilon', 'kernel', 'tol', 'max_iter']
        for key, value in params.items():
            if key in valid_params:
                regular_params[key] = value
            else:
                kernel_args_params[key] = value
        for key, value in regular_params.items():
            setattr(self, key, value)
        self.kernel_args.update(kernel_args_params)
        
        return self
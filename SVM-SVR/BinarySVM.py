import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score
from kernels import linear_kernel

class BinarySVM(BaseEstimator, ClassifierMixin):
    """
    Класс бинарного классификатора с помощью Supported Vector Machine.

    Для решения двойственной задачи используется Simplified SMO.

    The Simplified SMO Algorithm http://cs229.stanford.edu/materials/smo.pdf
    """
    def __init__(self, C=1.0, kernel=linear_kernel, tol=1e-3, max_iter=1000, **kernel_args):
        """
        Бинарный SVM с использованием Simplified SMO и поддержкой sklearn.GridSearchCV

        Args:
            C (float): Параметр регуляризации. Default = 1.0.
            kernel (sklearn or self-made): Ядро. Default = linear_kernel.
            tol (float): Точность сходимости. Default = 1e-3.
            max_iter (int): Число итераций. Default = 1000
            **kernel_args: Гиперпараметры ядра.
        Returns:
            Объект классификатора.
        """
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter
        self.kernel_args = kernel_args
        self.b = 0
        self.alpha = None
        self.n_samples = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None

    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray):
        """Метод вызова ядра"""
        if callable(self.kernel):
            return self.kernel(x1, x2, **self.kernel_args)
        else:
            raise ValueError(f"Неподдерживаемое ядро: {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Метод для "обучения" классификатора.

        Args:
            X (np.ndarray): Массив фичей.
            y (np.ndarray): Массив меток классов.
        Returns:
            "Обученный" классификатор.
        """
        X, y = check_X_y(X, y)
        self.n_samples = X.shape[0]
        
        self.K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                self.K[i, j] = self._kernel_function(X[i], X[j])
        
        self.alpha = np.zeros(self.n_samples)

        iter_count = 0
        while iter_count < self.max_iter:
            num_changed_alphas = 0
            for i in range(self.n_samples):
                E_i = np.sum(self.alpha * y * self.K[:, i]) + self.b - y[i]
                if ((y[i] * E_i < -self.tol and self.alpha[i] < self.C) or
                    (y[i] * E_i > self.tol and self.alpha[i] > 0)):
                    
                    j = i
                    while j == i:
                        j = np.random.randint(0, self.n_samples)
                    
                    E_j = np.sum(self.alpha * y * self.K[:, j]) + self.b - y[j]
                    
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue
                    
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    b1 = (self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] 
                          - y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j])
                    b2 = (self.b - E_j - y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j] 
                          - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j])
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                    
                    num_changed_alphas += 1

            iter_count += 1

        sv_mask = self.alpha > 1e-5
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self.support_vector_alphas = self.alpha[sv_mask]
        
        self.is_fitted_ = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Метод, который делает предсказания для массива фичей.

        Args:
            X (np.ndarray): Массив фичей.

        Returns:
            np.ndarray: w^T * x + b
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        kernel_matrix = np.zeros((X.shape[0], len(self.support_vectors)))
        for i in range(X.shape[0]):
            for j, sv in enumerate(self.support_vectors):
                kernel_matrix[i, j] = self._kernel_function(X[i], sv)
        
        decisions = np.dot(kernel_matrix, 
                        self.support_vector_alphas * self.support_vector_labels) + self.b
        return decisions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Метод, который делает предсказания для массива фичей.

        В отличие от decision_function(), возвращает метки классов в формате (1, -1).

        Args:
            X (np.ndarray): Массив фичей.

        Returns:
            np.ndarray: w^T * x + b
        """
        decisions = self.decision_function(X)
        return np.where(decisions >= 0, 1, -1)
    
    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """
        Какой-то метод для нахождения метрики. Видимо для выбора лучших гиперпараметров
        Args:
            X (np.ndarray): Массив фичей.
            y (np.ndarray): Массив меток класса.

        Returns:
            score (float):
            (Sklearn) returns the number of correctly classified samples (int).
        """
        return accuracy_score(y, self.predict(X))

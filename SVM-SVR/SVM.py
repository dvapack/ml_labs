import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from kernels import linear_kernel
from BinarySVM import BinarySVM

class SVM(BaseEstimator, ClassifierMixin):
    """
    Класс классификатора с помощью Support Vector Machine.

    Для решения двойственной задачи используется Simplified SMO.

    The Simplified SMO Algorithm http://cs229.stanford.edu/materials/smo.pdf
    """
    def __init__(self, C=1.0, kernel=linear_kernel, tol=1e-3, max_iter=1000, **kernel_args):
        """
        SVM с использованием Simplified SMO и поддержкой sklearn.GridSearchCV

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

        self.classifiers = []
        self.classes_ = None

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
        self.classes_ = unique_labels(y)
        
        for class_label in self.classes_:
            y_binary = np.where(y == class_label, 1, -1)
            classifier = BinarySVM(
                C=self.C, 
                kernel=self.kernel, 
                tol=self.tol, 
                max_iter=self.max_iter,
                **self.kernel_args
            )
            classifier.fit(X, y_binary)
            self.classifiers.append(classifier)
        
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
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        decision_scores = np.zeros((n_samples, n_classes))
        
        for i, classifier in enumerate(self.classifiers):
            decision_scores[:, i] = classifier.decision_function(X)
    
        predictions = self.classes_[np.argmax(decision_scores, axis=1)]
        return predictions
    
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

    def get_params(self, deep=True):
        """
        Метод для sklearn.GridSearchCV
        """
        params = super().get_params(deep=False)
        params.update({
            'C': self.C,
            'kernel': self.kernel,
            'tol': self.tol,
            'max_iter': self.max_iter
        })
        params.update(self.kernel_args)
        return params

    def set_params(self, **params):
        """
        Метод для sklearn.GridSearchCV
        """
        kernel_args_params = {}
        regular_params = {}
        for key, value in params.items():
            if key in ['C', 'kernel', 'tol', 'max_iter']:
                regular_params[key] = value
            else:
                kernel_args_params[key] = value
        for key, value in regular_params.items():
            setattr(self, key, value)
        if kernel_args_params:
            self.kernel_args.update(kernel_args_params)
        return self
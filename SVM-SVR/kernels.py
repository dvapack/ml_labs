import numpy as np
import scipy.spatial.distance as dist

def linear_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Линейное ядро.
    Args:
        a (np.ndarray): Массив чисел
        b (np.ndarray): Массив чисел
    Returns:
        ndarray: (Numpy docs) Returns the dot product of a and b. If a and b are both scalars or both 1-D arrays 
        then a scalar is returned; otherwise an array is returned. If out is given, then it is returned.
    """
    return np.dot(a, b.T)

def poly_kernel(a: np.ndarray, b: np.ndarray, degree: int=2,  coef_r: float=0.0):
    """
    Полиномиальное ядро.
    Args:
        a (np.ndarray): Массив чисел
        b (np.ndarray): Массив чисел
        degree (int): Степень полиномиальности
        coef_r (float): Коэффициент...
    Returns:
        ndarray: K(x, y) = (x * y + c)^d
    """
    return (np.dot(a, b) + coef_r)**degree

def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma):
    """
    Гауссово (rbf) ядро.
    Args:
        a (np.ndarray): Массив чисел
        b (np.ndarray): Массив чисел
        gamma: Коэффициент...
    Returns:
        ndarray: K(x, y) = exp(- gamma  ||x - y ||^2)
    """
    return np.exp(-gamma * dist.euclidean(a, b) ** 2)

def sigmoid_kernel(a: np.ndarray, b: np.ndarray, alpha: float, c: float):
    """
    Сигмоидальное ядро.
    Args:
        a (np.ndarray): Массив чисел
        b (np.ndarray): Массив чисел
        alpha (float): Коэффициент...
        c (float): Коэффициент...
    Returns:
        ndarray: K(x, y) = tanh( alpha x * y + c)
    """
    return np.tanh(alpha * np.dot(a, b) + c)
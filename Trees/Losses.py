import numpy as np

class Loss:        
    def base_predictions(self, y):
        pass

    def gradients(self, F_i, y_i) -> float:
        pass

    def hessians(self, F_i, y_i) -> float:
        pass

class MSELoss(Loss):
    def base_predictions(self, y):
        return np.mean(y)

    def gradients(self, F_i, y_i) -> float:
        return F_i - y_i
    
    def hessians(self, F_i, y_i):
        return np.ones_like(F_i)
    
class MAELoss(Loss):
    def base_predictions(self, y):
        return np.mean(y)

    def gradients(self, F_i, y_i):
        return F_i - y_i
    
    def hessians(self, F_i, y_i):
        return np.ones_like(F_i)
    
class LogisticLoss(Loss):
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def base_predictions(self, y):
        pos_ratio = np.mean(y)
        pos_ratio = np.clip(pos_ratio, 1e-15, 1 - 1e-15)
        return np.log(pos_ratio / (1 - pos_ratio))
    
    def gradients(self, F_i, y_i):
        p = self._sigmoid(F_i)
        return p - y_i
    
    def hessians(self, F_i, y_i):
        p = self._sigmoid(F_i)
        return p * (1 - p)

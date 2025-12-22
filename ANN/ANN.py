import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from activation_functions import sigmoid, tanh, relu, leaky_relu, \
                                    sigmoid_grad, tanh_grad, relu_grad, leaky_relu_grad


class MyMLP:
    def __init__(self, layer_sizes, activations, weight_init='xavier', 
                 optimizer='sgd', learning_rate=0.01, **opt_params):
        """
        Инициализация MLP
        
        Parameters:
        -----------
        layer_sizes : list
            Размеры слоев [input_size, hidden1_size, ..., output_size]
        activations : list
            Функции активации для каждого слоя (кроме входного)
        weight_init : str
            Метод инициализации весов: 'xavier' или 'he'
        optimizer : str
            Оптимизатор: 'sgd', 'momentum', 'rmsprop', 'adam'
        learning_rate : float
            Базовый learning rate
        **opt_params : dict
            Дополнительные параметры для оптимизаторов:
            - momentum: beta (momentum coefficient)
            - rmsprop: beta, epsilon
            - adam: beta1, beta2, epsilon
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weight_init = weight_init
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.opt_params = opt_params
        
        assert len(activations) == len(layer_sizes) - 1, \
            "Number of activations should be one less than number of layers"
        
        self.weights = []
        self.biases = []
        self._init_weights()
        
        self._init_optimizer_params()
        
        self.best_val_loss = float('inf')
        self.best_weights = None
        self.best_biases = None
        self.no_improvement_count = 0
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        self.use_batchnorm = opt_params.get('use_batchnorm', False)
        if self.use_batchnorm:
            self.gamma = [np.ones(size) for size in layer_sizes[1:-1]]
            self.beta = [np.zeros(size) for size in layer_sizes[1:-1]]
            self.running_mean = [np.zeros(size) for size in layer_sizes[1:-1]]
            self.running_var = [np.ones(size) for size in layer_sizes[1:-1]]
            self.bn_momentum = opt_params.get('bn_momentum', 0.9)
        
        self.dropout_rates = opt_params.get('dropout_rates', [0.0] * (len(layer_sizes) - 1))
        self.dropout_masks = []
        
        self.l2_lambda = opt_params.get('l2_lambda', 0.0)

    def _init_weights(self):
        """Инициализация весов в соответствии с выбранным методом"""
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            if self.weight_init == 'xavier':
                limit = np.sqrt(6 / (input_size + output_size))
                W = np.random.uniform(-limit, limit, (input_size, output_size))
            elif self.weight_init == 'he':
                std = np.sqrt(2 / input_size)
                W = np.random.normal(0, std, (input_size, output_size))
            else:
                W = np.random.randn(input_size, output_size) * 0.01
            
            b = np.zeros(output_size)
            
            self.weights.append(W)
            self.biases.append(b)

    def _init_optimizer_params(self):
        """Инициализация параметров для разных оптимизаторов"""
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        
        if self.optimizer in ['rmsprop', 'adam']:
            self.s_w = [np.zeros_like(w) for w in self.weights]
            self.s_b = [np.zeros_like(b) for b in self.biases]
        
        self.beta = self.opt_params.get('beta', 0.9)  # для momentum и RMSprop
        self.beta1 = self.opt_params.get('beta1', 0.9)  # для Adam
        self.beta2 = self.opt_params.get('beta2', 0.999)  # для Adam
        self.epsilon = self.opt_params.get('epsilon', 1e-8)  # для RMSprop и Adam

    def _activation(self, x, activation_type):
        """Выбор функции активации"""
        if activation_type == 'sigmoid':
            return sigmoid(x)
        elif activation_type == 'tanh':
            return tanh(x)
        elif activation_type == 'relu':
            return relu(x)
        elif activation_type == 'leaky_relu':
            alpha = self.opt_params.get('leaky_alpha', 0.01)
            return leaky_relu(x, alpha)
        else:
            raise ValueError(f"Unknown activation: {activation_type}")

    def _activation_grad(self, x, activation_type):
        """Градиент функции активации"""
        if activation_type == 'sigmoid':
            return sigmoid_grad(x)
        elif activation_type == 'tanh':
            return tanh_grad(x)
        elif activation_type == 'relu':
            return relu_grad(x)
        elif activation_type == 'leaky_relu':
            alpha = self.opt_params.get('leaky_alpha', 0.01)
            return leaky_relu_grad(x, alpha)
        else:
            raise ValueError(f"Unknown activation: {activation_type}")

    def _batch_norm_forward(self, x, layer_idx, training=True):
        """Прямой проход BatchNorm"""
        if not self.use_batchnorm or layer_idx >= len(self.layer_sizes) - 2:
            return x
            
        gamma = self.gamma[layer_idx]
        beta = self.beta[layer_idx]
        
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            self.running_mean[layer_idx] = self.bn_momentum * self.running_mean[layer_idx] + \
                                          (1 - self.bn_momentum) * batch_mean
            self.running_var[layer_idx] = self.bn_momentum * self.running_var[layer_idx] + \
                                         (1 - self.bn_momentum) * batch_var
            
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            out = gamma * x_norm + beta
            
            self.bn_cache.append({
                'x_norm': x_norm,
                'batch_mean': batch_mean,
                'batch_var': batch_var,
                'x_centered': x - batch_mean,
                'gamma': gamma
            })
        else:
            x_norm = (x - self.running_mean[layer_idx]) / np.sqrt(self.running_var[layer_idx] + self.epsilon)
            out = gamma * x_norm + beta
            
        return out

    def forward(self, X, training=True):
        """Прямой проход через сеть"""
        self.A = [X]  # activations
        self.Z = []   # pre-activations
        self.bn_cache = []
        
        A = X
        
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.Z.append(Z)
            
            if self.use_batchnorm and i < len(self.weights) - 1:
                Z = self._batch_norm_forward(Z, i, training)
            
            if i < len(self.weights) - 1:
                A = self._activation(Z, self.activations[i])
                if training and self.dropout_rates[i] > 0:
                    mask = (np.random.rand(*A.shape) > self.dropout_rates[i]) / (1 - self.dropout_rates[i])
                    A = A * mask
                    if i >= len(self.dropout_masks):
                        self.dropout_masks.append(mask)
                    else:
                        self.dropout_masks[i] = mask
                self.A.append(A)
            else:
                A = self._softmax(Z)
                self.A.append(A)
        
        return self.A[-1]

    def _softmax(self, x):
        """Softmax activation для последнего слоя"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _cross_entropy_loss(self, y_pred, y_true, epsilon=1e-15):
        """Cross-entropy loss с L2 регуляризацией"""
        n_samples = y_true.shape[0]
        
        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[np.arange(n_samples), y_true] = 1
        
        loss = -np.sum(y_true_onehot * np.log(np.clip(y_pred, epsilon, 1 - epsilon))) / n_samples
        
        l2_reg = 0.0
        if self.l2_lambda > 0:
            for w in self.weights:
                l2_reg += np.sum(w**2)
            l2_reg = self.l2_lambda * l2_reg / (2 * n_samples)
        
        return loss + l2_reg

    def backward(self, X, y):
        """Обратный проход (backpropagation)"""
        n_samples = X.shape[0]
        
        y_onehot = np.zeros((n_samples, self.layer_sizes[-1]))
        y_onehot[np.arange(n_samples), y] = 1
        
        delta = self.A[-1] - y_onehot
        
        dW = np.dot(self.A[-2].T, delta) / n_samples
        db = np.sum(delta, axis=0) / n_samples
        
        if self.l2_lambda > 0:
            dW += self.l2_lambda * self.weights[-1] / n_samples
        
        self.grads_w = [dW]
        self.grads_b = [db]
        
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(delta, self.weights[i + 1].T)
            
            if self.dropout_rates[i] > 0 and len(self.dropout_masks) > i:
                delta = delta * self.dropout_masks[i]
            
            activation_grad = self._activation_grad(self.Z[i], self.activations[i])
            delta = delta * activation_grad
            
            dW = np.dot(self.A[i].T, delta) / n_samples
            db = np.sum(delta, axis=0) / n_samples
            
            if self.l2_lambda > 0:
                dW += self.l2_lambda * self.weights[i] / n_samples
            
            self.grads_w.insert(0, dW)
            self.grads_b.insert(0, db)

    def update_weights(self):
        """Обновление весов с использованием выбранного оптимизатора"""
        for i in range(len(self.weights)):
            if self.optimizer == 'sgd':
                self.weights[i] -= self.learning_rate * self.grads_w[i]
                self.biases[i] -= self.learning_rate * self.grads_b[i]
                
            elif self.optimizer == 'momentum':
                self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * self.grads_w[i]
                self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * self.grads_b[i]
                
                self.weights[i] -= self.learning_rate * self.v_w[i]
                self.biases[i] -= self.learning_rate * self.v_b[i]
                
            elif self.optimizer == 'rmsprop':
                self.s_w[i] = self.beta * self.s_w[i] + (1 - self.beta) * self.grads_w[i]**2
                self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * self.grads_b[i]**2
                
                self.weights[i] -= self.learning_rate * self.grads_w[i] / (np.sqrt(self.s_w[i]) + self.epsilon)
                self.biases[i] -= self.learning_rate * self.grads_b[i] / (np.sqrt(self.s_b[i]) + self.epsilon)
                
            elif self.optimizer == 'adam':
                self.v_w[i] = self.beta1 * self.v_w[i] + (1 - self.beta1) * self.grads_w[i]
                self.v_b[i] = self.beta1 * self.v_b[i] + (1 - self.beta1) * self.grads_b[i]
                
                self.s_w[i] = self.beta2 * self.s_w[i] + (1 - self.beta2) * self.grads_w[i]**2
                self.s_b[i] = self.beta2 * self.s_b[i] + (1 - self.beta2) * self.grads_b[i]**2
                
                v_w_corr = self.v_w[i] / (1 - self.beta1**(self.t))
                v_b_corr = self.v_b[i] / (1 - self.beta1**(self.t))
                s_w_corr = self.s_w[i] / (1 - self.beta2**(self.t))
                s_b_corr = self.s_b[i] / (1 - self.beta2**(self.t))
                
                self.weights[i] -= self.learning_rate * v_w_corr / (np.sqrt(s_w_corr) + self.epsilon)
                self.biases[i] -= self.learning_rate * v_b_corr / (np.sqrt(s_b_corr) + self.epsilon)
                
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def fit(self, X, y, X_val=None, y_val=None, epochs=50, batch_size=64, 
            patience=5, lr_schedule=None):
        """
        Обучение модели
        
        Parameters:
        -----------
        X : array-like
            Обучающие данные
        y : array-like
            Метки обучающих данных
        X_val, y_val : array-like, optional
            Валидационные данные для early stopping
        epochs : int
            Количество эпох
        batch_size : int
            Размер мини-батча
        patience : int
            Количество эпох без улучшения для early stopping
        lr_schedule : callable, optional
            Функция для изменения learning rate
        """
        n_samples = X.shape[0]
        self.t = 0  # шаг для Adam
        
        for epoch in tqdm(range(epochs), desc='Training'):
            if lr_schedule is not None:
                self.learning_rate = lr_schedule(epoch)
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            epoch_acc = 0
            
            for i in range(0, n_samples, batch_size):
                self.t += 1  # увеличиваем шаг для Adam
                
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = self.forward(X_batch, training=True)
                
                self.backward(X_batch, y_batch)
                
                self.update_weights()
                
                batch_loss = self._cross_entropy_loss(y_pred, y_batch)
                batch_acc = accuracy_score(y_batch, np.argmax(y_pred, axis=1))
                
                epoch_loss += batch_loss * len(X_batch)
                epoch_acc += batch_acc * len(X_batch)
            
            epoch_loss /= n_samples
            epoch_acc /= n_samples
            
            self.train_losses.append(epoch_loss)
            self.train_accs.append(epoch_acc)
            
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val, training=False)
                val_loss = self._cross_entropy_loss(val_pred, y_val)
                val_acc = accuracy_score(y_val, np.argmax(val_pred, axis=1))
                
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_weights = [w.copy() for w in self.weights]
                    self.best_biases = [b.copy() for b in self.biases]
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                    if self.no_improvement_count >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        self.weights = [w.copy() for w in self.best_weights]
                        self.biases = [b.copy() for b in self.best_biases]
                        break
            
            if epoch % 1 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")
        
        if hasattr(self, 'best_weights') and self.best_weights is not None:
            self.weights = [w.copy() for w in self.best_weights]
            self.biases = [b.copy() for b in self.best_biases]

    def predict(self, X):
        """Предсказание меток классов"""
        y_pred = self.forward(X, training=False)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        """Предсказание вероятностей классов"""
        return self.forward(X, training=False)

    def plot_learning_curves(self):
        """Визуализация кривых обучения"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        if hasattr(self, 'val_losses') and len(self.val_losses) > 0:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Accuracy')
        if hasattr(self, 'val_accs') and len(self.val_accs) > 0:
            plt.plot(self.val_accs, label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def visualize_first_layer_weights(self, title='First Layer Weights'):
        """Визуализация весов первого слоя для MNIST"""
        if self.layer_sizes[0] != 784:
            print("This method is designed for MNIST (784 input features)")
            return
        
        W = self.weights[0]
        n_neurons = W.shape[1]
        grid_size = int(np.ceil(np.sqrt(n_neurons)))
        
        plt.figure(figsize=(12, 12))
        for i in range(n_neurons):
            plt.subplot(grid_size, grid_size, i + 1)
            neuron_weights = W[:, i].reshape(28, 28)
            plt.imshow(neuron_weights, cmap='viridis')
            plt.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
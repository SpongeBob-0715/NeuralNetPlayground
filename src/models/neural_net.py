import numpy as np

class ThreeLayerNet:
    def __init__(self, input_dim, hidden_size, output_dim, activation='relu', dropout_rate=0.5):
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.is_training = True
        
        self._initialize_params()

    def _initialize_params(self):
        self.params = {
            'W1': np.random.randn(self.input_dim, self.hidden_size) * np.sqrt(2. / self.input_dim),
            'b1': np.zeros(self.hidden_size),
            'W2': np.random.randn(self.hidden_size, self.output_dim) * np.sqrt(2. / self.hidden_size),
            'b2': np.zeros(self.output_dim)
        }

    def _activation_function(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, X):
        self.X = X
        self.z1 = np.dot(X, self.params['W1']) + self.params['b1']
        self.a1 = self._activation_function(self.z1)
        
        if self.is_training:
            self.dropout_mask = (np.random.rand(*self.a1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.a1 *= self.dropout_mask
        
        self.z2 = np.dot(self.a1, self.params['W2']) + self.params['b2']
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, y, reg):
        num_samples = self.X.shape[0]
        delta3 = self.probs.copy()
        delta3[range(num_samples), y] -= 1
        delta3 /= num_samples
        
        grads = {
            'W2': np.dot(self.a1.T, delta3) + reg * self.params['W2'],
            'b2': np.sum(delta3, axis=0),
        }
        
        delta2 = np.dot(delta3, self.params['W2'].T)
        if self.is_training:
            delta2 *= self.dropout_mask
        
        if self.activation == 'relu':
            delta2[self.z1 <= 0] = 0
        elif self.activation == 'sigmoid':
            delta2 *= self.a1 * (1 - self.a1)
        
        grads['W1'] = np.dot(self.X.T, delta2) + reg * self.params['W1']
        grads['b1'] = np.sum(delta2, axis=0)
        return grads

    def set_training(self, training=True):
        self.is_training = training
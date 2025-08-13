import numpy as np


class LogisticRegression:
	
	def __init__(self, lr=0.01, n_iter=1000):
		self.lr = lr
		self.n_iter = n_iter
		self.weights = None
		self.bias = None


	def _sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		for _ in range(self.n_iter):
			linear_model = np.dot(X, self.weights) + self.bias
			y_predict = self._sigmoid(linear_model)

			dw = (1 / n_samples) * np.dot(X.T, (y_predict - y))
			db = (1 / n_samples) * np.sum(y_predict - y)

			self.weights -= self.lr * dw
			self.bias -= self.lr * db

	def predict_prob(self, X):
		linear_model = np.dot(X, self.weights) + self.bias
		return self._sigmoid(linear_model)

	def predict(self, X):
		prob = self.predict_prob(X)
		return np.where(prob >= 0.5, 1, 0)
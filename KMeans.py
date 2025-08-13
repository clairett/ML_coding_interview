import numpy as np

class KMeans:
	
	def __init__(self, k, max_iteration, tol=1e-4):
		self.k = k
		self.max_iteration = max_iteration
		self.tol = tol

	def fit(self, X):
		n_samples, n_features = X.shape

		random_indice = np.random.choice(n_samples, self.k, replace=False)
		self.centroids = X[random_indice]
		for _ in range(self.max_iteration):
			distances = self._compute_distance(X)
			labels = np.argmin(distances, axis=1)
			print("labels:")
			print(labels)
			
			new_centroids = np.zeros_like(self.centroids)
			for i in range(self.k):
				points_in_cluster = X[labels == i]
				if len(points_in_cluster) > 0:
					new_centroids[i] = points_in_cluster.mean(axis=0)
				else:
					# handle the pitfall in kmeans
					idx = np.random.choice(n_samples)
					new_centroids[i] = X[idx]

			if np.all(np.linalg.norm(self.centroids-new_centroids, axis=1) < self.tol):
				break

			self.centroids = new_centroids

	def _compute_distance(self, X):
		"""
		X is a 2D array of shape (n_samples, n_features)
		self.centroids is a 2D array of shape (n_clusters, n_features)
		"""
		return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

	def predict(self, X):
		distances = self._compute_distance(X)
		return np.argmin(distances, axis=1)


if __name__ == "__main__":
	X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
	kmeans = KMeans(k=2, max_iteration=10)
	kmeans.fit(X)
	# print(kmeans.centroids)
	# print(kmeans.predict(X))
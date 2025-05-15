import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, x):
        X = np.array(x)
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # axis=1 tells NumPy: "operate across columns for each row
        distances = np.linalg.norm(self.X_train - x, axis=1)
        
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y[k_indices]
        most_common = Counter(k_labels).most_common(1)
        # the most_common return a tuple
        return most_common[0][0]

# Example usage:
if __name__ == "__main__":
    # Sample training data
    X_train = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
    y_train = [0, 0, 0, 1, 1]

    # Create and train classifier
    knn = KNearestNeighbors(k=3)
    knn.fit(X_train, y_train)

    # Sample test data
    X_test = [[2, 2], [6, 6]]
    predictions = knn.predict(X_test)

    print("Predictions:", predictions)

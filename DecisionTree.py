import numpy as np

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        gini = 1.0 - sum((count / len(y))**2 for count in counts)
        return gini

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_idx, best_thr = None, None
        n_features = X.shape[1]
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                gini = (len(left_mask) * gini_left + len(right_mask) * gini_right)/len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_idx = feature_index
                    best_thr = threshold
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = X[:, feature_idx] > threshold
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        return DecisionTreeNode(feature_idx, threshold, left, right)

    def _most_common_label(self, y):
        counts = np.bincount(y)
        return np.argmax(counts)






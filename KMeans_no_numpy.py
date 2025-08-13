import random
import math

class KMeans:
    def __init__(self, k, max_iterations=100) -> None:
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []
        
    def euclidean_distance(self, p1, p2):
        return math.sqrt(sum((x1 - x2)**2 for x1, x2 in zip(p1, p2)))

    def assign_to_centroids(self, data):
        clusters = [[] for _ in range(len(self.centroids))]
        for point in data:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            closet_centroid = distances.index(min(distances))
            clusters[closet_centroid].append(point)
        return clusters

    def update_centroids(self, clusters):
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                new_centroid = [sum(coord) / len(cluster) for coord in zip(*cluster)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice(data))
        return new_centroids


    def fit(self, data, k, max_iterations=100):	
        # random initialization
        self.centroids = random.sample(data, k)

        for _ in range(max_iterations):
            # step 1: assign points to nearest centroids
            clusters = self.assign_to_centroids(data, centroids)

            # step 2: update centroids
            new_centroids = self.update_centroids(clusters)

            # step 3: check convergence
            if new_centroids == centroids:
                break
            self.centroids = new_centroids

    def predict(self, x):
        predictions = []
        for point in X:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            closet_centroid = distances.index(min(distances))
            predictions.append(closet_centroid)
        return predictions
    



# Example usage
data = [
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0],
    [5, 2], [5, 4], [5, 0]
]

k = 2  # Number of clusters
centroids, clusters = k_means(data, k)

print("Centroids:", centroids)
print("Clusters:", clusters)
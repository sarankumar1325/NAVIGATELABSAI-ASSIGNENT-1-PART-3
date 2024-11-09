

---

# K-means Clustering Algorithm

## Table of Contents
- [Overview](#overview)
- [Introduction to K-means Clustering](#introduction-to-k-means-clustering)
- [Key Concepts and Terminology](#key-concepts-and-terminology)
- [Steps in K-means Clustering](#steps-in-k-means-clustering)
- [Advantages and Limitations](#advantages-and-limitations)
- [Practical Applications](#practical-applications)
- [Code Implementation](#code-implementation)
- [Results](#results)
- [Conclusion](#conclusion)

## Overview
This project demonstrates an implementation of the K-means clustering algorithm for unsupervised learning. It includes creating a sample dataset, performing K-means clustering both manually and using `scikit-learn`, and visualizing the results with cluster centroids. A silhouette score is also calculated to evaluate the clustering quality.

## Introduction to K-means Clustering
K-means clustering is an unsupervised learning algorithm that groups data points into \( k \) clusters by minimizing the variance within each cluster. The algorithm iteratively assigns points to clusters and adjusts centroids until convergence:
- **Centroids** are updated to the mean position of all points within each cluster.
- **Distance Metrics**: Euclidean distance is used to assign points to the nearest centroid.

## Key Concepts and Terminology
- **Centroid**: The center point of a cluster, calculated as the mean of all points in the cluster.
- **Inertia**: The sum of squared distances between each point and its nearest centroid.
- **Silhouette Score**: Measures clustering quality by comparing intra-cluster and inter-cluster distances.

## Steps in K-means Clustering
1. **Import Libraries**: Use `numpy`, `matplotlib`, and `sklearn`.
2. **Generate Dataset**: Create a simple dataset with points to be clustered.
3. **Initialize Centroids**: Randomly select points as initial centroids.
4. **Assign Points to Nearest Centroid**: Using Euclidean distance, assign each point to the closest centroid.
5. **Update Centroids**: Compute the mean of points within each cluster.
6. **Repeat Until Convergence**: Continue reassigning and updating until centroids stabilize.

## Advantages and Limitations
### Advantages
- **Simplicity**: K-means is computationally efficient and easy to understand.
- **Scalability**: Works well with large datasets.

### Limitations
- **Cluster Shape**: Assumes clusters are spherical and equally sized.
- **Initialization Sensitivity**: Different initial centroids can lead to different results.

## Practical Applications
- **Customer Segmentation**: Group customers with similar behaviors.
- **Image Compression**: Reduce color space in images.
- **Document Clustering**: Organize documents by topic.

## Code Implementation

### Libraries Used
- `numpy`: For numerical operations.
- `matplotlib`: For data visualization.
- `scikit-learn`: For implementing K-means and calculating silhouette score.

### Dataset
A small dataset with points like \((1,1)\), \((1.5,2)\), etc., is used to demonstrate clustering into two clusters.

### Sample Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate dataset
X = np.array([
    [1, 1],
    [1.5, 2],
    [3, 4],
    [5, 7],
    [3.5, 5],
    [4.5, 5],
    [3.5, 4.5]
])

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, init=np.array([[1, 1], [1.5, 2]]), n_init=1, random_state=42)
kmeans.fit(X)

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=100, marker='o', edgecolor='k', label='Data Points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title("K-means Clustering Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.show()

# Calculate silhouette score
sil_score = silhouette_score(X, kmeans.labels_)
print(f"Silhouette Score: {sil_score:.2f}")
```

## Results
The K-means clustering algorithm successfully grouped the points into two clusters. The final centroids and silhouette score indicate good clustering quality. The plot visually shows well-separated clusters with centroids marked in red.

## Conclusion
This K-means implementation demonstrates the fundamental principles of clustering. The algorithm effectively grouped data points, and the silhouette score provided an indication of clustering quality. Further exploration with more complex datasets and higher values of \( k \) can extend this analysis.

---

⭐ **Star this repository if you found it helpful!**

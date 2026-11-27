# GMM Source: https://towardsdatascience.com/gaussian-mixture-models-gmms-from-theory-to-implementation-4406c7fe9847/
import random
import numpy as np #Used for matrix multiplication and matrix initialization
from data.clean_data import get_data
from methods.kmeans import kmeans

NUM_OF_VALUES = 4

# Create a matrix that corresponds to their current label
# Returns centroid_probability as mean vector
# Returns label_matrix as covariance matrix
def covariance_matrix(X, final_centroids, labels, k):
    mean_vector = []
    label_matrix = []
    e_identity = np.eye(NUM_OF_VALUES, dtype = float) * 10**-6
    for i in range(0, k):
        label_counter = 0
        centroid_probability = 0
        clusters = []
        matrix_record = []
        current_matrix = np.zeros((NUM_OF_VALUES, NUM_OF_VALUES))
        for label in labels:
            if (label == i):
                label_counter += 1
                clusters.append(X[i])
        centroid_probability = label_counter / len(labels)
        # To be multiplied to current_matrix after getting sum of every new matrix
        cluster_spot = 1 / label_counter
        print("Iteration", i)
        print(centroid_probability)
        for cluster in clusters:
            for j in range(0, NUM_OF_VALUES):
                cluster[j] -= final_centroids[i][j]
            new_matrix = np.array([cluster])
            # Creates a 4 x 4 matrix using numpy by multiplying array to its transpose
            new_matrix = new_matrix.T @ new_matrix
            matrix_record.append(new_matrix)
        for matrix in matrix_record:
            current_matrix += matrix
        current_matrix = (current_matrix * cluster_spot) + e_identity
        mean_vector.append(centroid_probability)
        label_matrix.append(current_matrix)
    return mean_vector, label_matrix
        


def gaussian_mixture_model():
    name, X, features = get_data("data/dataset.csv")
    k = 3
    initial_centroids = random.sample(features,k)

    # Gather the initial mean vector using final_centroids from kmeans
    final_centroids, labels, count = kmeans(features, initial_centroids)
    covariance_matrix(X, final_centroids, labels, k)
    return 

# Set the probabilities of each data point

def expectation_stage():
    return

# Calculate the likelihood that a data point exists with a particular cluster
def maximization_stage():
    return

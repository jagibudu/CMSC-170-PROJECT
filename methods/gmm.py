# GMM Source: https://towardsdatascience.com/gaussian-mixture-models-gmms-from-theory-to-implementation-4406c7fe9847/
import random
import numpy as np #Used for matrix operations
import math #To get the exact value of pi, log and exp
from data.clean_data import get_data
from methods.kmeans import kmeans

#Claude was used to debug the gmm

NUM_OF_VALUES = 4
MAX_ITERATIONS = 100
TOLERANCE = 10**-6

# Create a matrix that corresponds to their current label
# Returns centroid_probability as weights
# Returns label_matrix as covariance matrix
def initial_covariance_matrix(X, final_centroids, labels, k):
    weights = []
    label_matrix = []
    e_identity = np.eye(NUM_OF_VALUES, dtype = float) * 10**-6
    for i in range(0, k):
        label_counter = 0
        clusters = []
        current_matrix = np.zeros((NUM_OF_VALUES, NUM_OF_VALUES))
        for j in range(0, len(labels)):
            if (labels[j] == i):
                label_counter += 1
                clusters.append(X[j])
        centroid_probability = label_counter / len(labels)
        for cluster in clusters:
            # Copy so the data itself is not changed
            diff = np.array([cluster]) - np.array([final_centroids[i]])
            # Creates a 4 x 4 matrix using numpy by multiplying array to its transpose
            current_matrix += diff.T @ diff
        if (label_counter != 0):
            current_matrix = (current_matrix / label_counter) + e_identity
        else:
            current_matrix = np.eye(NUM_OF_VALUES, dtype = float)
        weights.append(centroid_probability)
        label_matrix.append(current_matrix)
    mean_vector = [list(centroid) for centroid in final_centroids]
    return mean_vector, weights, label_matrix

def gaussian_mixture_model(k = 3):
    name, X, features = get_data("data/dataset.csv")
    random.seed(10)
    initial_centroids = random.sample(features, k)

    # Gather the initial mean vector using final_centroids from kmeans
    final_centroids, labels, count = kmeans(features, initial_centroids)

    # Initialize mean vector, weight, and covariance matrix after k-means
    mean_vector, weights, label_matrix = initial_covariance_matrix(features, final_centroids, labels, k)

    old_likelihood = None
    iteration = 1
    while (iteration <= MAX_ITERATIONS):
        # Find the responsibility
        responsibility, likelihood = expectation_stage(features, mean_vector, weights, label_matrix, k)

        # Update the weights, mean vectors and covariance matrices
        mean_vector, weights, label_matrix = maximization_stage(features, mean_vector, weights, label_matrix, responsibility, k)

        # Stop when the log-likelihood barely changes
        if (old_likelihood != None and abs(likelihood - old_likelihood) < TOLERANCE):
            break
        old_likelihood = likelihood
        iteration += 1

    # Each province goes to the cluster with the highest responsibility
    responsibility, likelihood = expectation_stage(features, mean_vector, weights, label_matrix, k)
    gmm_labels = []
    for row in responsibility:
        gmm_labels.append(row.index(max(row)))
    return gmm_labels

# Set the probabilities of each data point
def expectation_stage(X, mean_vector, weights, label_matrix, k):
    # Every cluster has a fixed determinant and inverse
    # To be used for every responsibility of X
    determinants = []
    inverses = []
    for i in range(0, k):
        determinants.append(np.linalg.det(label_matrix[i]))
        inverses.append(np.linalg.inv(label_matrix[i]))

    data_responsibility = []
    likelihood = 0
    for data in X:
        log_values = []
        for i in range(0, k):
            if (weights[i] == 0):
                log_values.append(-math.inf)
                continue
            diff = np.array(data) - np.array(mean_vector[i])
            # Squared Mahalanobis Distance
            smd = diff @ inverses[i] @ diff
            # Log of the density formula, so small numbers do not become 0
            log_density = -1/2 * smd - (NUM_OF_VALUES/2) * math.log(2 * math.pi) - 1/2 * math.log(determinants[i])
            log_values.append(math.log(weights[i]) + log_density)

        # Subtract the biggest value before exp so it does not become 0
        biggest = max(log_values)
        current_responsibility = []
        for value in log_values:
            current_responsibility.append(math.exp(value - biggest))
        total = sum(current_responsibility)
        # Responsibilities of one province add up to 1
        for i in range(0, k):
            current_responsibility[i] = current_responsibility[i] / total
        data_responsibility.append(current_responsibility)
        likelihood += biggest + math.log(total)
    return data_responsibility, likelihood

# Calculate the likelihood that a data point exists with a particular cluster
def maximization_stage(X, mean_vector, weights, label_matrix, responsibility, k):
    e_identity = np.eye(NUM_OF_VALUES, dtype = float) * 10**-6
    new_mean_vector = []
    new_weights = []
    new_label_matrix = []
    for i in range(0, k):
        responsibility_sum = 0
        for j in range(0, len(X)):
            responsibility_sum += responsibility[j][i]

        # Keep the old values if the cluster is empty
        if (responsibility_sum == 0):
            new_weights.append(0)
            new_mean_vector.append(mean_vector[i])
            new_label_matrix.append(label_matrix[i])
            continue

        # Creates new weight
        new_weights.append(responsibility_sum / len(X))

        # Creates new mean vector
        mat = np.zeros(NUM_OF_VALUES)
        for j in range(0, len(X)):
            mat = mat + responsibility[j][i] * np.array(X[j])
        mat = mat / responsibility_sum
        new_mean_vector.append(list(mat))

        # Creates new covariance matrix
        current_matrix = np.zeros((NUM_OF_VALUES, NUM_OF_VALUES))
        for j in range(0, len(X)):
            diff = np.array([X[j]]) - np.array([mat])
            current_matrix += responsibility[j][i] * (diff.T @ diff)
        current_matrix = (current_matrix / responsibility_sum) + e_identity
        new_label_matrix.append(current_matrix)
    return new_mean_vector, new_weights, new_label_matrix
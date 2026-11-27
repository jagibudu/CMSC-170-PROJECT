# GMM Source: https://towardsdatascience.com/gaussian-mixture-models-gmms-from-theory-to-implementation-4406c7fe9847/
import random
import numpy as np #Used for matrix operations
import math #To get the exact value of e and pi
from decimal import Decimal
from data.clean_data import get_data
from methods.kmeans import kmeans

NUM_OF_VALUES = 4

# Create a matrix that corresponds to their current label
# Returns centroid_probability as weights
# Returns label_matrix as covariance matrix
def initial_covariance_matrix(X, final_centroids, labels, k):
    weights = []
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
        print("Mean Vector #", i)
        print("Initial weight:", centroid_probability)
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
        weights.append(centroid_probability)
        label_matrix.append(current_matrix)
    return final_centroids, weights, label_matrix

def gaussian_mixture_model():
    name, X, features = get_data("data/dataset.csv")
    k = 3
    iteration = 1
    initial_centroids = random.sample(features,k)

    # Gather the initial mean vector using final_centroids from kmeans
    final_centroids, labels, count = kmeans(features, initial_centroids)

    # Initialize mean vector, weight, and covariance matrix after k-means
    mean_vector, weights, label_matrix = initial_covariance_matrix(X, final_centroids, labels, k)

    while(iteration != 10):
        print("Iteration:", iteration)
    # Find the responsibility
        responsibility = expectation_stage(X, mean_vector, weights, label_matrix, k, iteration)

    # Update the weights
        mean_vector, weights, label_matrix = maximization_stage(X, mean_vector, weights, labels, responsibility, k)
        iteration += 1

    print(mean_vector, weights, label_matrix)
    return 

# Set the probabilities of each data point
def expectation_stage(X, final_centroids, weights, label_matrix, k, iteration):
    # Every final centroid has a fixed set of determinant and inverse
    # To be used for every responsibility of X
    determinants = []
    inverses = []
    if (iteration == 1):
        div = 10e13
    else:
        div = 1
    for i in range(0, k):
        determinants.append(np.linalg.det(label_matrix[i]))
        inverses.append(np.linalg.inv(label_matrix[i]))

    data_density = []
    data_responsibility = []
    e = math.e
    pi = math.pi
    counter = 0
    for data in X:
        current_density = []
        current_responsibility = []
        
        for i in range(0, k):
            smd = 0
            density_nominator = 0
            density_denominator = 0
            for j in range(0, NUM_OF_VALUES):
                data[j] -= final_centroids[i][j]
            one_matrix = np.array([data])
            # Squared Mahanabolis Distance
            smd = one_matrix @ inverses[i] @ one_matrix.T
            # Density formula
            # Probability that X happens on current cluster
            # Current number is too low without dividing by 10^13
            density_nominator = Decimal(1 * e ** (-1/2 * smd[0][0] / div))
            density_denominator = Decimal((2 * pi)**(NUM_OF_VALUES/2) * (determinants[i] ** 1/2))
            density = float(density_nominator / density_denominator)
            current_density.append(density)
            current_responsibility.append(density * weights[i])
        data_density.append(current_density)
        data_responsibility.append(current_responsibility)
        counter += 1
    return data_responsibility
        
# Calculate the likelihood that a data point exists with a particular cluster
def maximization_stage(X, mean_vector, weights, labels, responsibility, k):
    responsibility_sum = []
    label_matrix = []
    for i in range(0, k):
        responsibility_sum.append(0)
    for y in responsibility:
        for i in range(0, k):
            responsibility_sum[i] = responsibility_sum[i] + y[i]
    # Creates new weight
    for i in range(0, k):
        if (responsibility_sum[i] == 0):
            weights[i] = 0
        else:
            weights[i] = responsibility_sum[i] / len(X)
    # Creates new mean vector
    for i in range(0, k):
        mat = np.zeros((NUM_OF_VALUES, 1))
        for j in range(0, len(X)):
            if (labels[j] == i):
                current_mat = np.array([X[j]])
                add_mat = current_mat.T
                mat = mat + add_mat
        if (responsibility_sum[i] != 0):
            mat = mat * (1/responsibility_sum[i])
            mean_vector[i] = [float(mat[0][0]), float(mat[1][0]), float(mat[2][0]), float(mat[3][0])]
        else:
            mean_vector[i] = 0
    # Creates new covariance matrix
    for i in range(0, k):
        current_matrix = np.zeros((NUM_OF_VALUES, NUM_OF_VALUES))
        clusters = []
        matrix_record = []
        counter = 0
        e_identity = np.eye(NUM_OF_VALUES, dtype = float) * 10**-6
        for j in range(0, len(X)):
            if (labels[j] == i):
                clusters.append(X[j])
                counter += 1
        for cluster in clusters:
            for j in range(0, NUM_OF_VALUES):
                cluster[j] = cluster[j] - mean_vector[i][j]
            new_matrix = np.array([cluster])
            new_matrix = new_matrix.T @ new_matrix
            matrix_record.append(new_matrix)
        for matrix in matrix_record:
            current_matrix += matrix
        if (counter != 0):
            current_matrix = (current_matrix * 1/counter) + e_identity
        else:
            current_matrix = np.zeros((NUM_OF_VALUES, NUM_OF_VALUES))
        label_matrix.append(current_matrix)
    return mean_vector, weights, label_matrix

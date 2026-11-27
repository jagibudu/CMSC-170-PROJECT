# GMM Source: https://towardsdatascience.com/gaussian-mixture-models-gmms-from-theory-to-implementation-4406c7fe9847/
import random
import numpy as np #Used for matrix operations
import math #To get the exact value of e and pi
from decimal import Decimal
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

    # Initialize mean vector and covariance matrix after k-means
    mean_vector, label_matrix = covariance_matrix(X, final_centroids, labels, k)

    # Find the responsibility
    expectation_stage(X, final_centroids, mean_vector, label_matrix, k)
    return 

# Set the probabilities of each data point
def expectation_stage(X, final_centroids, mean_vector, label_matrix, k):
    # Every final centroid has a fixed set of determinant and inverse
    # To be used for every responsibility of X
    determinants = []
    inverses = []
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
            density_nominator = Decimal(1 * e ** (-1/2 * smd[0][0] / 10e13))
            density_denominator = Decimal((2 * pi)**(NUM_OF_VALUES/2) * (determinants[i] ** 1/2))
            density = float(density_nominator / density_denominator)
            current_density.append(density)
            current_responsibility.append(density * mean_vector[i])
        data_density.append(current_density)
        data_responsibility.append(current_responsibility)
        print("Data Point#", counter + 1)
        print("Densities: ", current_density)
        print("Responsibilities: ", current_responsibility)
        counter += 1
    return data_density, data_responsibility
        
# Calculate the likelihood that a data point exists with a particular cluster
def maximization_stage():
    return

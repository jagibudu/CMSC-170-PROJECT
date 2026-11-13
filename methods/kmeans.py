import random
from data.preprocess import load_data
from data.clean_data import clean_records, build_features

# Gets the cleaned and scaled features from preprocess.py, returns names(locations),features
def read_data():
    time_columns, records = load_data("data/dataset.csv")
    clean = clean_records(time_columns, records)
    scaled_rows = build_features(time_columns, clean)
 
    names = []
    features = []
    for row in scaled_rows:
        names.append(row["location"])
        features.append(row["features"])
    return names, features
 
# Computes the Euclidean distance between a feature and a centroid
def euclidean(feature,initial_centroid):
    distance = 0
    for i in range(len(feature)):
        distance += (feature[i] - initial_centroid[i]) ** 2
    return distance ** (1/2)
 
 
# Update centroids by averaging the corresponding coordinates of the feature vectors
def update_centroids(features,labels,centroids):
    k = len(centroids)
    dims = len(features[0])
    # give each cluster a slot
    sums = []
    for j in range(k):
        sums.append([0.0] * dims)
    count = [0] * k  # one slot per cluster, e.g. k = 3 [0.0, 0.0, 0.0].
 
    # Add each point to the totals of its cluster and counts the total instances of a cluster
    for i in range(len(features)):
        label = labels[i]
        for d in range(dims):
            sums[label][d] += features[i][d]
        count[label] += 1
 
    new_centroids = []
    # Get the averages of the totals for each cluster to form the new centroids
    for j in range(k):
        if count[j] == 0:
            # Empty cluster, keep its old centroid.
            new_centroids.append(centroids[j])
        else:
            centroid = []
            for d in range(dims):
                centroid.append(sums[j][d] / count[j])
            new_centroids.append(centroid)
 
    return new_centroids
 
# kmeans algorithm
def kmeans(features,centroids):
 
    # count the number of iterations
    counter = 0
 
    # Loop until the centroids no longer change
    while 1:
 
        # Store classifications
        labels = []
 
        # Loop over all the features
        for j in range(len(features)):
 
            # Initialize a distance list for every feature
            distances = []
 
            # Get the distance from this feature to each of the k centroids
            for i in range(len(centroids)):
                distance = euclidean(features[j],centroids[i])
                distances.append(distance)
 
            # Get the smallest distance to any centroid
            min_distance = min(distances)
 
            # gets the index of the cluster(cluster0,cluster1,etc.)
            label = distances.index(min_distance)
            labels.append(label)
 
        new_centroids = update_centroids(features,labels,centroids)
 
        counter+=1
        # Break out of the while loop if it converged
        if new_centroids == centroids:
            break
 
        # Update the centroids
        centroids = new_centroids
 
    return centroids,labels,counter


def runkmeans():
    names,features = read_data()

    # number of clusters
    k = 3

    # k random features as the initial centroids
    initial_centroids = random.sample(features,k)

    final_centroids,labels,count = kmeans(features,initial_centroids)

    print("K-Means Clustering Output from dataset.csv")
    print(f"k = {k}")
    print(f"Iterations: {count}")
    print()

    # Cluster sizes: how many areas are in each cluster
    print("Cluster sizes:")
    for i in range(k):
        print(f"Cluster {i}: {labels.count(i)} areas")
    print()

    # Cluster profiles: each final centroid is the average of the features of its areas
    crops = ["Irrigated palay", "Rainfed palay", "White corn", "Yellow corn"]
    for i in range(k):
        centroid = final_centroids[i]
        print(f"Cluster {i} profile:")
        for j in range(len(crops)):
            print(f"  {crops[j]:<16} harvest share Q1-Q4: {centroid[j*4]:.2f} {centroid[j*4+1]:.2f} {centroid[j*4+2]:.2f} {centroid[j*4+3]:.2f}")
        print(f"  Irrigated share of palay: {centroid[16]:.2f}")
        print(f"  Yellow share of corn:     {centroid[17]:.2f}")
        print(f"  Palay share of total:     {centroid[18]:.2f}")
        print()

    # Print points grouped by cluster
    print("Labeled Dataset:")
    for i in range(k):
        for j in range(len(names)):
            if labels[j] == i:
                print(f"{labels[j]} : {names[j]}")


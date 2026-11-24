import random
from data.clean_data import get_data

#References: Lab Handout 07/Lab Exer 07

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
    # names: (region, province), X: tons, features: standardized (what we cluster on)
    names,X,features = get_data("data/dataset.csv")

    # number of clusters
    k = 3

    # add a seed so the results can be reproducible
    random.seed(10)

    # k random features as the initial centroids
    initial_centroids = random.sample(features,k)

    final_centroids,labels,count = kmeans(features,initial_centroids)

    print("K-Means Clustering Output from dataset.csv")
    print(f"k = {k}")
    print(f"Iterations: {count}")
    print()

    # Cluster sizes: how many provinces are in each cluster
    print("Cluster sizes:")
    for i in range(k):
        print(f"Cluster {i}: {labels.count(i)} provinces")
    print()

    # Cluster profiles: average tons per year of the provinces in each cluster
    crops = ["Irrigated palay", "Rainfed palay", "White corn", "Yellow corn"]
    for i in range(k):
        # get the rows (in tons) of the provinces in this cluster
        members = []
        for j in range(len(X)):
            if labels[j] == i:
                members.append(X[j])

        print(f"Cluster {i} profile (average tons per year):")
        if len(members) == 0:
            print("  empty cluster")
            print()
            continue
        for c in range(len(crops)):
            total = 0.0
            for row in members:
                total += row[c]
            print(f"  {crops[c]:<16} {total / len(members):>12,.0f}")
        print()

    # Print points grouped by cluster
    print("Labeled Dataset:")
    for i in range(k):
        for j in range(len(names)):
            if labels[j] == i:
                region, province = names[j]
                print(f"{labels[j]} : {province} ({region})")
    
    return labels
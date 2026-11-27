

# calculates the euclidean distance between two points
def euclidean(a, b):
    total = 0.0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return total ** 0.5


#Reference: https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index
#Calinski-Harabasz index, 0 up to infinity (higher is better)
#It measures how clearly the provinces split into separate groups.
#High score: Provinces in the same group are very similar, and each group is clearly different from the others.
#Low score: Provinces are mixed together, making it hard to tell one group from another.
#Pass the feature matrix (Z) and the labels produced by a method to use
def calinski_harabasz(Z, labels):
    #n:number of data points(provinces),k:number of clusters
    n = len(Z)
    k = max(labels) + 1
    # number of features
    dims = len(Z[0])

    #overall data centroid (mean)
    mean = [0.0] * dims   # make a slot for each feature: [0.0, 0.0, 0.0, 0.0]
    for i in range(n):
        for j in range(dims): # go through each feature of province i
            mean[j] += Z[i][j]
    # get the average of each crop across all provinces
    for j in range(dims):
        mean[j] = mean[j] / n

    # centroid of each cluster: average each feature using only the provinces in that cluster
    # count keeps how many provinces are in each cluster
    centroids = []
    for i in range(k):
        centroids.append([0.0] * dims)
    count = [0] * k
    for i in range(n):
        for j in range(dims):
            centroids[labels[i]][j] += Z[i][j]
        count[labels[i]] += 1
    for i in range(k):
        for j in range(dims):
            centroids[i][j] = centroids[i][j] / count[i]

    #B = Between-Cluster Sum of Squares: squared distance of each centroid to the overall mean,
    #times the number of points in that cluster
    B = 0
    for i in range(k):
        B += count[i] * euclidean(centroids[i], mean) ** 2

    #W = Within-Cluster Sum of Squares: squared distance of each point to its own centroid
    W = 0
    for i in range(n):
        W += euclidean(Z[i], centroids[labels[i]]) ** 2

    return (B/(k - 1)) / (W/(n - k))



 
def silhouette_score(X, labels):
    n = len(X)  # Number of data points (provinces)

    # Get all unique cluster labels
    clusters = sorted(set(labels))

    # requires at least 2 clusters, but fewer clusters than data points
    if len(clusters) < 2 or len(clusters) >= n:
        raise ValueError("Invalid number of clusters.")

    # Sum of silhouette scores for all provinces
    total_score = 0  

    # Calculate silhouette score for each province
    for i in range(n):
        own_cluster = labels[i]  # Cluster where the current province belongs

  
        # Calculate a(i)
        same_distances = []

        for j in range(n):
            # Skip the current province and only consider provinces in the same cluster
            if labels[j] == own_cluster and i != j:
                distance = euclidean(X[i], X[j])  # Calculate Euclidean distance
                same_distances.append(distance)

        # If the province is alone in its cluster (singleton cluster),
        # its silhouette score is defined as 0
        if len(same_distances) == 0:
            continue

        # Compute the average distance within the same cluster
        a = sum(same_distances) / len(same_distances)

        # Calculate b(i)
        b = float("inf")  # Initialize b to infinity so any actual distance is smaller

        for cluster in clusters:
            # Ignore the province's own cluster
            if cluster == own_cluster:
                continue

            other_distances = []  # Distances to provinces in another cluster

            for j in range(n):
                # Get distances only to provinces belonging to this other cluster
                if labels[j] == cluster:
                    distance = euclidean(X[i], X[j])
                    other_distances.append(distance)

            # Calculate the average distance to this other cluster
            average_distance = sum(other_distances) / len(other_distances)

            # Keep the smallest average distance (nearest neighboring cluster)
            if average_distance < b:
                b = average_distance

        # Calculate silhouette score

        # Prevent division by zero if both distances are zero
        if max(a, b) == 0:
            score = 0
        else:
            score = (b - a) / max(a, b)

        # Add the current province's score to the total
        total_score += score

    # Return the average silhouette score of all provinces
    return total_score / n
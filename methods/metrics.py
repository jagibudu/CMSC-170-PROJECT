

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
from data.clean_data import get_data

FEATURE_CROPS = [
    "Irrigated Palay",
    "Rainfed Palay",
    "White Corn",
    "Yellow Corn"
]


# Calculate Euclidean distance
def euclidean_distance(a, b):
    squared_difference = 0.0

    for i in range(len(a)):
        squared_difference += (a[i] - b[i]) ** 2

    return squared_difference ** 0.5


# Calculate average linkage between two clusters
def average_linkage(cluster_a, cluster_b, X):
    total_distance = 0.0
    pair_count = 0

    for i in cluster_a:
        for j in cluster_b:
            total_distance += euclidean_distance(X[i], X[j])
            pair_count += 1

    return total_distance / pair_count


# Perform agglomerative hierarchical clustering
def agglomerative_clustering(X, n_clusters=3):
    num_samples = len(X)

    if num_samples == 0:
        raise ValueError("No observations provided.")

    if not 1 <= n_clusters <= num_samples:
        raise ValueError("Invalid number of clusters.")

    # Start with each province in its own cluster
    clusters = [[i] for i in range(num_samples)]

    merge_history = []

    # Merge clusters until the desired number remains
    while len(clusters) > n_clusters:
        min_distance = float("inf")

        merge_a = -1
        merge_b = -1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = average_linkage(
                    clusters[i],
                    clusters[j],
                    X
                )

                if distance < min_distance:
                    min_distance = distance
                    merge_a = i
                    merge_b = j

        if merge_a == -1 or merge_b == -1:
            raise RuntimeError("Unable to find clusters to merge.")

        # Save the merge information
        merge_history.append({
            "cluster_a": clusters[merge_a][:],
            "cluster_b": clusters[merge_b][:],
            "distance": min_distance
        })

        # Combine the two clusters
        merged_cluster = (
            clusters[merge_a] + clusters[merge_b]
        )

        clusters[merge_a] = merged_cluster
        clusters.pop(merge_b)

    return clusters, merge_history


# Generate cluster labels
def get_labels(clusters, num_samples):
    labels = [-1] * num_samples

    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id

    return labels


# Run hierarchical clustering
def run_hierarchicalClustering(filepath="data/dataset.csv", n_clusters=3):

    # Get data from the shared preprocessing pipeline
    province_names, X, X_scaled = get_data(filepath)

    if len(X_scaled) < n_clusters:
        raise ValueError("Not enough complete provinces.")

    # Perform clustering
    clusters, merge_history = agglomerative_clustering(
        X_scaled,
        n_clusters
    )

    # Generate cluster labels
    labels = get_labels(clusters, len(X_scaled))

    # Display results
    print("\nHIERARCHICAL CLUSTERING")
    print("Features:", FEATURE_CROPS)

    print("\nNumber of provinces:", len(province_names))
    print("Number of clusters:", n_clusters)

    print("\nCluster Assignments:")

    for cluster_id, cluster in enumerate(clusters):
        print(f"\nCluster {cluster_id + 1}:")

        for index in cluster:
            region, province = province_names[index]
            print(f"  {province} ({region})")

    print("\nMerge History:")

    for step, merge in enumerate(merge_history, start=1):
        print(
            f"Step {step}: "
            f"{merge['cluster_a']} + "
            f"{merge['cluster_b']} "
            f"at distance {merge['distance']:.4f}"
        )

    return {
        "province_names": province_names,
        "features": X,
        "standardized_features": X_scaled,
        "labels": labels,
        "clusters": clusters,
        "merge_history": merge_history
    }


if __name__ == "__main__":
    run_hierarchicalClustering("data/dataset.csv", 3)
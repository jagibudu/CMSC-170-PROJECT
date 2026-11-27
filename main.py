from data.clean_data import get_data
from methods.gmm import gaussian_mixture_model
from methods.kmeans import runkmeans
from methods.hierarchical import run_hierarchicalClustering
from methods.metrics import calinski_harabasz
from plots import plot_results

#Test methods
names, X, Z = get_data("data/dataset.csv")
labels = runkmeans()
result = run_hierarchicalClustering("data/dataset.csv", 3)
labels2 = result["labels"]
labels3 = gaussian_mixture_model()

ch_kmeans = calinski_harabasz(Z, labels)
ch_hierarchical = calinski_harabasz(Z, labels2)
ch_gmm = calinski_harabasz(Z, labels3)

print("EVALUATION RESULTS OF METHODS")
print("KMEANS")
print("Calinski-Harabasz: ", ch_kmeans)


print("Hierarchical")
print("Calinski-Harabasz: ", ch_hierarchical)

print("Gaussian Mixture Model")
print("Calinski-Harabasz: ", ch_gmm)

# Plot the results (saved in figures/)
plot_results(
    names,
    X,
    {"K-Means": labels, "Hierarchical": labels2, "GMM": labels3},
    {"K-Means": ch_kmeans, "Hierarchical": ch_hierarchical, "GMM": ch_gmm},
)

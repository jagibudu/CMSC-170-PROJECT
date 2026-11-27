from data.clean_data import get_data
from methods.kmeans import runkmeans
from methods.hierarchical import run_hierarchicalClustering
from methods.gmm import gaussian_mixture_model
from methods.metrics import calinski_harabasz, silhouette_score
from plots import plot_results

#Test methods
names, X, Z = get_data("data/dataset.csv")
labels = runkmeans()
result = run_hierarchicalClustering("data/dataset.csv", 3)
labels2 = result["labels"]
gmm_labels = gaussian_mixture_model()

results = {"K-Means": labels, "Hierarchical": labels2, "GMM": gmm_labels}

# Silhouette and Calinski-Harabasz of each method
silhouette = {}
ch = {}
for method in results:
    silhouette[method] = silhouette_score(Z, results[method])
    ch[method] = calinski_harabasz(Z, results[method])

print("EVALUATION RESULTS OF METHODS")
for method in results:
    print(method)
    print("Silhouette: ", silhouette[method])
    print("Calinski-Harabasz: ", ch[method])

# Plot the results (saved in figures/)
plot_results(
    names,
    X,
    results,
    {"Silhouette": silhouette, "Calinski-Harabasz": ch},
)

from data.clean_data import get_data
from methods.kmeans import runkmeans
from methods.hierarchical import run_hierarchicalClustering
from methods.metrics import calinski_harabasz

#Test methods
names, X, Z = get_data("data/dataset.csv")
labels = runkmeans()
result = run_hierarchicalClustering("data/dataset.csv", 3)
labels2 = result["labels"]


print("EVALUATION RESULTS OF METHODS")
print("KMEANS")
print("Calinski-Harabasz: ", (calinski_harabasz(Z, labels)))


print("Hierarchical")
print("Calinski-Harabasz: ", (calinski_harabasz(Z, labels2)))

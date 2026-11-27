import os
import matplotlib.pyplot as plt

CROPS = ["Irrigated Palay", "Rainfed Palay", "White Corn", "Yellow Corn"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
FIGURES_DIR = "figures"


# scatter of the provinces on two crops (fx and fy), colored by cluster
def plot_scatter(ax, names, X, labels, fx, fy, title):
    k = max(labels) + 1

    for c in range(k):
        xs = []
        ys = []
        for i in range(len(X)):
            if labels[i] == c:
                xs.append(X[i][fx])
                ys.append(X[i][fy])
        ax.scatter(xs, ys, color=COLORS[c], label=f"Cluster {c + 1} ({len(xs)})",
                   edgecolors="black")

    # label the 3 biggest producers
    totals = []
    for i in range(len(X)):
        totals.append((X[i][fx] + X[i][fy], i))
    totals.sort(reverse=True)
    for total, i in totals[:3]:
        region, province = names[i]
        ax.annotate(province, (X[i][fx], X[i][fy]), fontsize=8)

    # symlog since a few provinces are way bigger than the rest and some have 0 tons
    ax.set_xscale("symlog", linthresh=100)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel(f"{CROPS[fx]} (avg tons/year)")
    ax.set_ylabel(f"{CROPS[fy]} (avg tons/year)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# average tons per year of each crop, one bar per cluster
def plot_profiles(ax, X, labels, title):
    k = max(labels) + 1
    width = 0.8 / k

    for c in range(k):
        members = []
        for i in range(len(X)):
            if labels[i] == c:
                members.append(X[i])
        if len(members) == 0:
            continue

        averages = []
        for f in range(len(CROPS)):
            total = 0.0
            for row in members:
                total += row[f]
            averages.append(total / len(members) / 1000)

        # shift the bars so the clusters sit side by side
        positions = []
        for f in range(len(CROPS)):
            positions.append(f - 0.4 + width * c + width / 2)

        ax.bar(positions, averages, width, color=COLORS[c],
               label=f"Cluster {c + 1} ({len(members)})")

    ax.set_xticks(range(len(CROPS)))
    ax.set_xticklabels(CROPS)
    ax.set_ylabel("Avg production (thousand tons/year)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


# score of each method for one metric
def plot_scores(ax, scores, metric):
    methods = []
    values = []
    for method in scores:
        methods.append(method)
        values.append(scores[method])

    bars = ax.bar(methods, values, color=COLORS[:len(methods)])
    ax.bar_label(bars, fmt="%.3f")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} score (higher is better)")
    ax.grid(True, axis="y", alpha=0.3)


# results: {method name: labels}, scores: {metric name: {method name: score}}
def plot_results(names, X, results, scores):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    n = len(results)

    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    row = 0
    for method in results:
        labels = results[method]
        plot_scatter(axes[row][0], names, X, labels, 0, 1, f"{method}: Palay")
        plot_scatter(axes[row][1], names, X, labels, 2, 3, f"{method}: Corn")
        row += 1
    fig.tight_layout()
    fig.savefig(FIGURES_DIR + "/clusters_scatter.png")

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    col = 0
    for method in results:
        plot_profiles(axes[col], X, results[method], f"{method}: cluster profiles")
        col += 1
    fig.tight_layout()
    fig.savefig(FIGURES_DIR + "/cluster_profiles.png")

    # one bar chart per metric, side by side
    m = len(scores)
    fig, axes = plt.subplots(1, m, figsize=(6 * m, 5))
    col = 0
    for metric in scores:
        plot_scores(axes[col], scores[metric], metric)
        col += 1
    fig.tight_layout()
    fig.savefig(FIGURES_DIR + "/evaluation_metrics.png")

    print("\nSaved figures to figures/")
    plt.show()

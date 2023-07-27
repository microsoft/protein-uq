import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

from train_all import split_dict
from utils import load_and_scale_data


def create_parser():
    parser = argparse.ArgumentParser(description='Train and test dimensionality reduction models')
    parser.add_argument("--split", type=str)
    parser.add_argument('--method', type=str, choices=['pca', 'umap', 'tsne', 'pca_tsne', 'umap_tsne'], default='pca', help='Dimensionality reduction model')
    parser.add_argument("--representation", choices=["ohe", "esm"], type=str)
    parser.add_argument('--n_neighbors', type=int, default=15, help='Number of neighbors')
    parser.add_argument('--perplexity', type=float, default=30.0, help='Perplexity')
    parser.add_argument('--min_dist', type=float, default=0.1, help='Minimum distance')
    parser.add_argument('--metric', type=str, choices=['euclidean', 'manhattan', 'cosine', 'correlation', 'jaccard'], default='euclidean', help='Distance metric')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    return parser


def dimred_train_test(split, representation, method, n_neighbors=15, perplexity=30.0, min_dist=0.1, metric='euclidean', n_jobs=1, random_state=42):
    split_orig = split
    split = split_dict[split]
    dataset = re.findall(r"(\w*)\_", split_orig)[0]

    # Load data
    (
        _,
        _,
        _,
        _,
        x_scaler,
        y_scaler,
        train_seq,
        train_target,
        test_seq,
        test_target,
    ) = load_and_scale_data(
        representation, "gp", dataset, split, True, False, False, False, True
    )

    # combine train and test
    all_seq = np.concatenate((train_seq, test_seq))
    all_target = np.concatenate((train_target, test_target))

    # Do dimensionality reduction
    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=random_state)
        X = pca.fit_transform(all_seq)
        # X_train = pca.fit_transform(train_seq)
        # X_test = pca.transform(test_seq)

    elif method == "umap":
        import umap

        umap_ = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        X = umap_.fit_transform(all_seq)
        # X_train = umap_.fit_transform(train_seq)
        # X_test = umap_.transform(test_seq)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            metric=metric,
            random_state=random_state,
        )
        # t-SNE can't fit_transform on train and then transform on test
        X = tsne.fit_transform(all_seq)

    elif method == "pca_tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        pca = PCA(n_components=50, random_state=random_state)
        # X_test = pca.fit_transform(test_seq)
        X = pca.fit_tranform(all_seq)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            metric=metric,
            random_state=random_state,
        )
        # t-SNE can't fit_transform on train and then transform on test
        X = tsne.fit_transform(X)

    elif method == "umap_tsne":
        import umap
        from sklearn.manifold import TSNE

        umap_ = umap.UMAP(
            n_components=50,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        # X_test = umap_.fit_transform(test_seq)
        X = umap_.fit_transform(all_seq)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            metric=metric,
            random_state=random_state,
        )
        # t-SNE can't fit_transform on train and then transform on test
        # X_test = tsne.fit_transform(test_seq)
        X = tsne.fit_transform(X)

    # Make plots

    # if "tsne" not in method:
    #     plt.scatter(X_train[:, 0], X_train[:, 1], c='k', alpha=0.1, label="Train")
    # plt.scatter(X_test[:, 0], X_test[:, 1], c='r', alpha=0.1, label="Test")
    plt.scatter(X[:len(train_target), 0], X[:len(train_target), 1], c='k', alpha=0.1, label="Train")
    plt.scatter(X[len(train_target):, 0], X[len(train_target):, 1], c='r', alpha=0.1, label="Test")
    plt.xlabel("Reduced Dimension 1")
    plt.ylabel("Reduced Dimension 2")
    plt.title(f"{method} on {dataset} {split} {representation}")
    plt.grid(False)
    plt.axis('off')
    plt.legend()
    plt.savefig(f"dimred/new_{method}_train_vs_test_{dataset}_{split}_{representation}.pdf")
    plt.clf()

    # if "tsne" not in method:
    #     plt.scatter(X_train[:, 0], X_train[:, 1], c=train_target, cmap="viridis")
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=test_target, cmap="viridis")
    target_plot = plt.scatter(X[:, 0], X[:, 1], c=all_target, cmap="viridis")
    plt.xlabel("Reduced Dimension 1")
    plt.ylabel("Reduced Dimension 2")
    cb = plt.colorbar(target_plot, label="Target Value")
    cb.set_ticks(['min', 'max'])
    plt.title(f"{method} on {dataset} {split} {representation}")
    plt.grid(False)
    plt.axis('off')
    plt.savefig(f"dimred/new_{method}_train_and_test_color_by_prop_{dataset}_{split}_{representation}.pdf")
    cb.remove()
    plt.clf()

    return


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    dimred_train_test(args.split, args.representation, args.method, args.n_neighbors, args.perplexity, args.min_dist, args.metric, args.n_jobs, args.random_state)

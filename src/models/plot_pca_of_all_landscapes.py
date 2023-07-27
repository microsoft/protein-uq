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


def dimred_all(split, representation, method, n_neighbors=15, perplexity=30.0, min_dist=0.1, metric='euclidean', n_jobs=1, random_state=42):
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
    print(split, all_seq.shape, all_target.shape)

    # Do dimensionality reduction
    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=random_state)
        # X_train = pca.fit_transform(train_seq)
        # X_test = pca.transform(test_seq)
        X = pca.fit_transform(all_seq)
        # print PCA explained variance
        print(f"PCA explained variance ratio (2): {pca.explained_variance_ratio_}")
        print(f"PCA explained variance ratio sum (2): {sum(pca.explained_variance_ratio_)}")

    elif method == "umap":
        import umap

        umap_ = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        # X_train = umap_.fit_transform(train_seq)
        # X_test = umap_.transform(test_seq)
        X = umap_.fit_transform(all_seq)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init='pca',
            learning_rate='auto',
            metric=metric,
            random_state=random_state,
        )
        # X_test = tsne.fit_transform(test_seq)
        X = tsne.fit_transform(all_seq)

    elif method == "pca_tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        pca = PCA(n_components=50, random_state=random_state)
        # X_test = pca.fit_transform(test_seq)
        X = pca.fit_transform(all_seq)
        print(f"PCA explained variance ratio (50): {pca.explained_variance_ratio_}")
        print(f"PCA explained variance ratio sum (50): {sum(pca.explained_variance_ratio_)}")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init='pca',
            learning_rate='auto',
            metric=metric,
            random_state=random_state,
        )
        # X_test = tsne.fit_transform(test_seq)
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
            init='pca',
            learning_rate='auto',
            metric=metric,
            random_state=random_state,
        )
        # X_test = tsne.fit_transform(test_seq)
        X = tsne.fit_transform(X)

    plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.1)
    plt.xlabel("Reduced Dimension 1")
    plt.ylabel("Reduced Dimension 2")
    plt.title(f"{method} {dataset} {split} {representation}")
    plt.grid(False)
    plt.axis('off')
    plt.savefig(f"dimred_all/{method}_{dataset}_{split}_{representation}_{perplexity}.png")
    plt.clf()

    # save dimred embeddings
    np.save(f"dimred_all/{method}_{dataset}_{split}_{representation}_{perplexity}.npy", X[:, :2])

    return


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    dimred_all(args.split, args.representation, args.method, args.n_neighbors, args.perplexity, args.min_dist, args.metric, args.n_jobs, args.random_state)

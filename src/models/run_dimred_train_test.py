import dimred_train_test

for split in ['gb1_1', 'gb1_2', 'gb1_3', 'gb1_4']:
    for representation in ['esm', 'ohe']:
        for method in ['pca', 'umap', 'tsne', 'pca_tsne', 'umap_tsne']:
            dimred_train_test.main(split, method)
            print(f"Finished {split} {method}")

for split in ['meltome_1', 'aav_2', 'aav_5', 'aav_7']:
    for representation in ['esm', 'ohe']:
        for method in ['pca', 'umap']:
            dimred_train_test.main(split, method)
            print(f"Finished {split} {method}")

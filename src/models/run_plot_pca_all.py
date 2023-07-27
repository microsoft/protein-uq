from plot_pca_of_all_landscapes import dimred_all

for split in ['gb1_2']:  # ['gb1_4', 'aav_7', 'meltome_1']:
    for representation in ['esm', 'ohe']:
        for method in ['pca', 'pca_tsne']:
            for perplexity in [5, 30]:
                dimred_all(split, representation, method, perplexity=perplexity)
                print(f"Finished {split} {representation} {method}")

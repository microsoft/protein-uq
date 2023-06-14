from dimred_train_test import dimred_train_test

for method in ['pca', 'umap']:
    for split in ['gb1_1', 'gb1_2', 'gb1_3', 'gb1_4']:
        for representation in ['ohe', 'esm']:    
            dimred_train_test(split, representation, method)
            print(f"Finished {split} {representation} {method}")

for method in ['pca', 'umap']:
    for split in ['aav_2', 'aav_5', 'aav_7', 'meltome_1']:
        for representation in ['ohe', 'esm']:
            dimred_train_test(split, representation, method)
            print(f"Finished {split} {representation} {method}")

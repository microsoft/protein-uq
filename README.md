# Protein UQ
[//]: # (Badges)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7839141.svg)](https://doi.org/10.5281/zenodo.7839141)

Benchmark for uncertainty quantification (UQ) in protein engineering.

## Citation
If you use this code, please cite the following [manuscript](https://doi.org/10.1371/journal.pcbi.1012639):
```
@article{protein-uq,
  title={Benchmarking uncertainty quantification for protein engineering},
  author={Greenman, Kevin P and Amini, Ava P and Yang, Kevin K},
  journal={PLOS Computational Biology},
  volume={21},
  number={1},
  pages={e1012639},
  year={2025},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

## Reproducing Results from the Manuscript

### Environment Installation
1. Install [Anaconda or Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html) if you have not yet done so.
2. `conda install -c conda-forge mamba` (optional, but recommended, if you do not already have `mamba` installed)
3. `git clone git@github.com:microsoft/protein-uq.git`
4. `cd protein-uq`
5. `mamba env create -f environment.yml` or `conda env create -f environment.yml` (`mamba` is recommended for faster installation)
6. `conda activate protein-uq`

### Prepare Data in protein-uq repo
1. `cd data`
2. `for d in {aav,gb1,meltome}; do (cd $d; unzip splits.zip); done`

### Creating ESM Embeddings
We used the [FLIP](https://github.com/J-SNACKKB/FLIP) repository to generate ESM embeddings for our models. The following commands can be used to reproduce the embeddings used in this work:

0. `cd ..` (to leave the `protein-uq` directory)
1. `git clone --recurse-submodules git@github.com:J-SNACKKB/FLIP.git` (`--recurse-submodules` is required to clone the ESM submodule of the FLIP repo)
2. `cd splits`
3. `for d in {aav,gb1,meltome}; do (cd $d; unzip splits.zip); done`
4. `cd ../baselines`
5. `wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt` (this file is 7.3 GB)
6. `wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt`
6. `flip_esm_embedding_commands.sh` - This script contains the commands used to generate the ESM embeddings with train-val-test splits for the 8 tasks used in this work. These commands should be run from the `baselines/` directory of the FLIP repository, and the `protein-uq` conda env must be activated. Each command will take a while.

The embeddings will be saved in the `FLIP/baselines/embeddings/` directory. Pre-computed embeddings for the AAV landscape can also be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.6549368).

### Training and Evaluating Models with Uncertainty Quantification
A list of commands to perform all training and inference for our models in series is provided in `src/models/train_all_commands_series.sh`. The following is an example command:

```
python train_all.py --split gb1_1 --model ridge --representation ohe --uncertainty ridge --dropout 0.0 --scale --seed 0
```

In practice, we used the [LLMapReduce](https://supercloud.mit.edu/submitting-jobs#llmapreduce) command on the [MIT SuperCloud](https://supercloud.mit.edu/) to make the most efficient use of resources using the cluster's scheduler and run our jobs in parallel. Original `LLMapReduce` commands are provided in `src/models/LLMapReduce_commands.txt`.

### Active Learning
A list of commands to perform all of our active learning experiments is provided in `src/active_learning/active_learning_commands_series.sh`. The following is an example command:

```
python active_learning.py --split gb1_1 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy random --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
```

In practice, we used the [LLMapReduce](https://supercloud.mit.edu/submitting-jobs#llmapreduce) command on the [MIT SuperCloud](https://supercloud.mit.edu/) to make the most efficient use of resources using the cluster's scheduler and run our jobs in parallel. Original `LLMapReduce` commands are provided in `src/active_learning/LLMapReduce_commands.txt`.

### Bayesian Optimization
A list of commands to perform all of our Bayesian optimization experiments is provided in `src/active_learning/bo_commands_series.sh`. Bayesian optimization uses the same script as active learning, but with a different set of acquisition strategies. The following is an example command:

```
python active_learning.py --split gb1_4 --model ridge --representation esm --uncertainty ridge --scale --num_folds 3 --al_strategy score_greedy --num_al_loops 5 --al_topk 100 --mean --dropout 0.0
```

In practice, we used the [LLMapReduce](https://supercloud.mit.edu/submitting-jobs#llmapreduce) command on the [MIT SuperCloud](https://supercloud.mit.edu/) to make the most efficient use of resources using the cluster's scheduler and run our jobs in parallel. Original `LLMapReduce` commands are provided in `src/active_learning/LLMapReduce_commands.txt`.

### Plotting Results
The following notebooks provided in the `notebooks/` directory can be used to reproduce the figures and tables in the manuscript:
* `plot_results_1.ipynb`: Figures 2 and 3, Supplementary Figures A, B, D; Supplementary Tables A-AR
* `plot_results_2.ipynb`: Figure 4, Supplementary Figure C
* `plot_results_active_learning.ipynb`: Figure 5, Supplementary Figures E-BH
* `plot_results_bo.ipynb`: Figure 6 

These notebooks require output files in the `src/models/results/` and `src/active_learning/al_results/` directories, which can be reproduced using the commands above.

### Example Notebook
The Jupyter notebook at `notebooks/example.ipynb` provides an example of how to train a model and make predictions with uncertainty quantification.

## Support
Open bug reports and ask questions on [GitHub issues](https://github.com/microsoft/protein-uq/issues). See [SUPPORT](https://github.com/microsoft/protein-uq/blob/main/SUPPORT.md) for details.

## License
This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/microsoft/protein-uq/blob/main/LICENSE) for additional details.

## Contributing
See [CONTRIBUTING](https://github.com/microsoft/protein-uq/blob/main/CONTRIBUTING.md).

## Security
See [SECURITY](https://github.com/microsoft/protein-uq/blob/main/SECURITY.md).

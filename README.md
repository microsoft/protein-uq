# Protein UQ
[//]: # (Badges)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7839142.svg)](https://doi.org/10.5281/zenodo.7839142)

Benchmark for uncertainty quantification (UQ) in protein engineering.

## Citation
If you use this code, please cite the following manuscript:
```
@article{protein-uq,
  title={Benchmarking Uncertainty Quantification for Protein Engineering},
  author={Greenman, Kevin P. and Amini, Ava P. and Yang, Kevin K.},
  journal={TBD},
  doi={TBD},
  year={2023}
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
All training and inference for our models was done on the [MIT SuperCloud](https://supercloud.mit.edu/). We used this cluster's [LLMapReduce](https://supercloud.mit.edu/submitting-jobs#llmapreduce) command to make the most efficient use of resources using the cluster's scheduler and run our jobs in parallel. Original commands (run from `src/models/`):
```
LLMapReduce --mapper=mapper_ohe.sh --input=inputs_ohe.txt --output=output_ohe --gpuNameCount=volta:1 --np [4,2,20] --keep=True
LLMapReduce --mapper=mapper_esm.sh --input=inputs_esm.txt --output=output_esm --gpuNameCount=volta:1 --np [4,2,20] --keep=True
```
An equivalent list of commands in series is provided in `src/models/train_all_commands_series.sh`.

### Active Learning
All active learning for our models was done on the [MIT SuperCloud](https://supercloud.mit.edu/). We used this cluster's [LLMapReduce](https://supercloud.mit.edu/submitting-jobs#llmapreduce) command to make the most efficient use of resources using the cluster's scheduler and run our jobs in parallel. Scripts for running the jobs in series are also provided. Original command (run from `src/active_learning/`):
```
LLMapReduce --mapper=mapper.sh --input=inputs.txt --output=output --gpuNameCount=volta:1 --np [4,2,20] --keep=True
```

An equivalent list of commands in series is provided in `src/active_learning/active_learning_commands_series.sh`.

### Plotting Results
The following notebooks provided in the `notebooks/` directory can be used to reproduce the figures and tables in the manuscript:
* `plot_results_1.ipynb`: Figures 2, 3, S1, S2, S4; Tables S1-S22
* `plot_results_2.ipynb`: Figures 4, S3
* `plot_results_active_learning.ipynb`: Figures 5, S5-S57

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

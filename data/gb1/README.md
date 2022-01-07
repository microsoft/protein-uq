### Dataset description

The GB1 "four" variations set stems from a [2016 publication](https://elifesciences.org/articles/16965) in which mutations at four sites (V39, D40, G41 and V54) were probed against a binding assay. 
The full WT GB1 sequence was never included in the dataset, so it was inferred from side chain A of PDB's [5LDE](https://www.rcsb.org/structure/5LDE).

### Full dataset

The `four_mutations_full_data.csv` is an extension of the [supplement to the 2016 manuscript](https://doi.org/10.7554/eLife.16965.024). It contains the following extra columns w.r.t. the supplement:
- `sequence`: the "full" sequence of [5LDE](https://www.rcsb.org/structure/5LDE) with substitutions at sites V39, D40, G41 and V54.
- the train/test splits for different splits detailed below.


### Splits

All splits are regression splits on the `Fittness` value reported by the [2016 publication](https://elifesciences.org/articles/16965) in the [supplement](https://doi.org/10.7554/eLife.16965.024).
Train/Test splits are done as follows.

Splits ([semaphore legend](../../README.md#split-semaphore)):
- 🟢 `one_vs_rest`: `train` is wild type and all single mutations, `test` is everything else.
- 🟢 `two_vs_rest`: `train` is wild type, all single & double mutations, `test` is everything else.
- 🟢 `three_vs_rest`: `train` is wild type, all single, double & triple mutations, `test` is everything else.
- 🟢 `low_vs_high`: `train` are sequences with fitness below WT, `test` is everything else.
- 🟠 `sampled`: Randomly split sequences into `train`/`test` with 80/20% probability.


All splits are contained in the `splits.zip` file. These are CSV with colums:

- `sequence`: the AA sequence. May contain special characters!
- `target`: the prediction target, which is float value (the split is a regression).
- `set`: either `train` or `test`, if the sequence should be used for training or testing your model!


### Cite
From the publisher as Bibtex:
```
@article {10.7554/eLife.16965,
article_type = {journal},
title = {Adaptation in protein fitness landscapes is facilitated by indirect paths},
author = {Wu, Nicholas C and Dai, Lei and Olson, C Anders and Lloyd-Smith, James O and Sun, Ren},
editor = {Neher, Richard A},
volume = 5,
year = 2016,
month = {jul},
pub_date = {2016-07-08},
pages = {e16965},
citation = {eLife 2016;5:e16965},
doi = {10.7554/eLife.16965},
url = {https://doi.org/10.7554/eLife.16965},
abstract = {The structure of fitness landscapes is critical for understanding adaptive protein evolution. Previous empirical studies on fitness landscapes were confined to either the neighborhood around the wild type sequence, involving mostly single and double mutants, or a combinatorially complete subgraph involving only two amino acids at each site. In reality, the dimensionality of protein sequence space is higher (20\textsuperscript{\textit{L}}) and there may be higher-order interactions among more than two sites. Here we experimentally characterized the fitness landscape of four sites in protein GB1, containing 20\textsuperscript{4} = 160,000 variants. We found that while reciprocal sign epistasis blocked many direct paths of adaptation, such evolutionary traps could be circumvented by indirect paths through genotype space involving gain and subsequent loss of mutations. These indirect paths alleviate the constraint on adaptive protein evolution, suggesting that the heretofore neglected dimensions of sequence space may change our views on how proteins evolve.},
keywords = {saturation mutagenesis, deep sequencing, fitness landscape, epistasis, adaptive evolution},
journal = {eLife},
issn = {2050-084X},
publisher = {eLife Sciences Publications, Ltd},
}
```

### Data licensing

The RWA data downloaded from elife is subject to [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Modified data available in this repository and in the `splits` falls under [AFL-3](https://opensource.org/licenses/AFL-3.0).

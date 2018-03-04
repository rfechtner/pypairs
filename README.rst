====
PyPairs - Cell cycle phase prediction
====

This is package is a Python-reimplementation of the Pairs algorithm described by
A. Scialdone et al. (2015). Original paper available under:
<https://doi.org/10.1016/j.ymeth.2015.06.021>.

The algorithm aims to predict the cell cycle phase for samples based on their
trascriptome. It can be applied to bulk and single cell RNA data. The algorithm
consists of two parts: *sandbag* and *cyclone*

sandbag
----
This function implements the training step of the pair-based prediction method.
Pairs of genes (A, B) are identified from a training data set, with known cell
cycle phase for each sample. In each pair, the fraction of cells in phase G1
with expression of A > B (based on expression values in the dataset) and the
fraction with B > A in each other phase exceeds a set threshold fraction.
These pairs are defined as the marker pairs for G1. This is repeated for each
phase to obtain a separate marker pair set.

cyclone
----
This function implements the classification step. To illustrate, consider
classification of cells into G1 phase. Pairs of marker genes are identified with
sandbag, where the expression of the first gene in the training data is greater
than the second in G1 phase but less than the second in all other phases. For
each cell, cyclone calculates the proportion of all marker pairs where the
expression of the first gene is greater than the second in the new data. A high
proportion suggests that the cell is likely to belong in G1 phase, as the
expression ranking in the new data is consistent with that in the training data.
Proportions are not directly comparable between phases due to the use of
different sets of gene pairs for each phase. Instead, proportions are converted
into scores (see below) that account for the size and precision of the
proportion estimate. The same process is repeated for all phases, using the
corresponding set of marker pairs in pairs. Cells with G1 or G2M scores above
0.5 are assigned to the G1 or G2M phases, respectively.
(If both are above 0.5, the higher score is used for assignment.)
Cells can be assigned to S phase based on the S score, but a more reliable
approach is to define S phase cells as those with G1 and G2M scores below 0.5.

Example
----
`import pypairs
from pathlib import Path
import pandas

gencounts_training = pandas.read_csv(Path("./path/to/expression/matrix.csv"))
gencounts_training.set_index("Unnamed: 0", inplace=True)

# Index or labels
is_G1 = [0,1,2,3]
is_S = ["Sample4","Sample5","Sample6"]
is_G2M = [
  gencounts_training.columns.get_loc(c)
  for c in gencounts_training.columns 
  if "G2M" in c
]

annotation = {
  "G1": list(is_G1),
  "S": list(is_S),
  "G2M": list(is_G2M)
}

marker_pairs = pypairs.sandbag(
  gencounts_training, phases=annotation,
  fraction=0.65, processes=10, verboose=True
)

gencounts_test = pandas.read_csv(Path("./path/to/expression/matrix.csv"))
gencounts_test.set_index("Unnamed: 0", inplace=True)

prediction = pypairs.cyclone(
  gencounts_test, marker_pairs=marker_pairs,
  verboose=True, processes=5
)

print(prediction)`

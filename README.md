# PyPairs - A python scRNA-Seq classifier

This is a python-reimplementation of the _Pairs_ algorithm as described by A. Scialdone et. al. (2015).
Original Paper available under: https://doi.org/10.1016/j.ymeth.2015.06.021

The algorithm aims to classify single cells based on their transcriptomic signal. Initially created to predict cell
cycle phase from scRNA-Seq data, this algorithm can be used for various applications.

It is a supervised maschine learning algorithm and as such it consits of two components: 
training (sandbag) and prediction (cyclone)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes. 

### Installation

This package is hosted at PyPi (https://pypi.org/project/pypairs/) and can be installed on any system running Python3 
with:

```
python3 -m pip install pypairs
```

Alternatively, pypairs can be installed using [Conda](https://conda.io/docs/) (most easily obtained via the [Miniconda Python distribution](https://conda.io/miniconda.html)):

```
conda install -c bioconda pypairs
```

### Minimal example

Assuming you have two scRNA count files (csv, columns = samples, rows = genes) and one annotation file (csv, no header, 
two rows: "gene, class") a minimal example would look like this:

```
from pypairs import wrapper

trainings_matrix = [PATH TO MATRIX]
annotation = [PATH TO ANNOTATION]
testing_matrix = [PATH TO MATRIX]

marker_pairs = wrapper.sandbag_from_file(trainings_matrix, annotation)

prediction = wrapper.cyclone_from_file(testing_matrix, marker_pairs)
```

## Core Dependencis

* [Numpy](http://www.numpy.org/) 
* [Numba](https://numba.pydata.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scanpy](https://github.com/theislab/scanpy)

## Authors

* **Antonio Scialdone** - *original algorithm*
* **Ron Fechtner** - *implementation and extension in Python*

## License 

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

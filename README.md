# Requirements
Tested with:
* NVIDIA driver 440.33.01
* CUDA version 10.2

To run:
* get conda
* get [chemprop](https://github.com/chemprop/chemprop)
* `conda env create -f environment.yml`
* `conda activate chemprop`
* unzip their data.tar.gz also
* `python featurization.py data/tox21.csv > tox21.txt`
* `./main < tox21.txt`

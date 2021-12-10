# mprod_package

[![Build and test](https://github.com/UriaMorP/mprod_package/actions/workflows/build.yaml/badge.svg)](https://github.com/UriaMorP/mprod_package/actions/workflows/build.yaml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mprod-package)
[![Documentation Status](https://readthedocs.org/projects/mprod-package/badge/?version=latest)](https://mprod-package.readthedocs.io/en/latest/?badge=latest)
![Conda](https://img.shields.io/conda/dn/conda-forge/mprod-package?label=downloads%28conda-forge%29)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/mprod-package.svg)](https://anaconda.org/conda-forge/mprod-package)



Software implementation for tensor-tensor m-product framework [[1]](#1).
The library currently contains tubal QR and tSVDM decompositions, and the TCAM method for dimensionality reduction.


<p align="center">
  <img width="80%",height="80%",  src="https://user-images.githubusercontent.com/16097812/143407367-36c30aa4-da1f-4a8b-93db-470114486064.png" />
</p>

## Installation 

### Conda
The `mprod-package` is hosted in [conda-forge](https://conda-forge.org/) channel. 

```
conda install -c conda-forge mprod-package
```

### pip
```
pip install mprod-package 
```
See `mprod-package`s [pypi entry](https://pypi.org/project/mprod-package/)

### From source 
Make sure that all dependencies listed below are installed in and found in your path. 

Dependencies:
* python                    3.6.8
* scipy                     1.5.3
* scikit-learn              0.24.1
* numpy                     1.19.2
* dataclasses               0.7   (Only for python version < 3.7)
* pip                       21.0.1

Clone the repository, then from the package directory, run
```
pip install -e .
```

The above dependencies are stated with exact versions used for locally test `mprod-package`, these packages were obtained from conda-forge channel.


## References
<a id="1">[1]</a> 
Misha E. Kilmer, Lior Horesh, Haim Avron, and Elizabeth Newman.  Tensor-tensor algebra for optimal representation and compression of multiway data. Proceedings of the National Academy of Sciences, 118(28):e2015851118, jul 2021.

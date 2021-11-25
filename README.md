# mprod_package
Software implementation for tensor-tensor m-product framework [[1]](#1).
The library currently contains tubal QR and tSVDM decompositions, and the TCAM method for dimensionality reduction.


<p align="center">
  <img width="80%",height="80%",  src="https://user-images.githubusercontent.com/16097812/143407367-36c30aa4-da1f-4a8b-93db-470114486064.png" />
</p>

## Installation 
Make sure that all dependencies are installed in a newly created conda environment using the conda-forge channel

Dependencies:
* python                    3.6.8
* scipy                     1.5.3
* scikit-learn              0.24.1
* numpy                     1.19.2
* dataclasses               0.7
* pip                       21.0.1


Clone the repository, then from the package directory, run
```
pip install -e .
```




The package 



## References
<a id="1">[1]</a> 
Misha E. Kilmer, Lior Horesh, Haim Avron, and Elizabeth Newman.  Tensor-tensor algebra for optimal representation and compression of multiway data. Proceedings of the National Academy of Sciences, 118(28):e2015851118, jul 2021.

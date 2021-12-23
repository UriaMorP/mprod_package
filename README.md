# mprod_package

[![Build and test](https://github.com/UriaMorP/mprod_package/actions/workflows/build.yaml/badge.svg)](https://github.com/UriaMorP/mprod_package/actions/workflows/build.yaml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mprod-package)
[![Documentation Status](https://readthedocs.org/projects/mprod-package/badge/?version=latest)](https://mprod-package.readthedocs.io/en/latest/?badge=latest)
![Conda](https://img.shields.io/conda/dn/conda-forge/mprod-package?label=downloads%28conda-forge%29)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/mprod-package.svg)](https://anaconda.org/conda-forge/mprod-package)
[![Pypi Downloads](https://img.shields.io/pypi/dm/mprod-package.svg?label=Pypi%20downloads)](
https://pypi.org/project/mprod-package/)


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

* Make sure that all dependencies listed in `requirements.txt` file are installed . 
* Clone the repository, then from the package directory, run
```
pip install -e .
```

The dependencies in `requirements.txt` are stated with exact versions used for locally test `mprod-package`, these packages were obtained from conda-forge channel.

```python
import pandas as pd

file_path = "https://raw.githubusercontent.com/UriaMorP/" \
            "tcam_analysis_notebooks/main/Schirmer2018/Schirmer2018.tsv"

data_table = pd.read_csv(file_path, index_col=[0,1], sep="\t"
                       , dtype={'Week':int})
data_table = data_table.loc[:,data_table.median() > 1e-7]
data_table.rename(columns= {k:f"Fature_{e+1}" for e,k in enumerate(data_table.columns)}, inplace=True) 
data_table.shape

%matplotlib inline
```

## How to use `TCAM`

Given with a `pandas.DataFrame` of the data as below, with 2-level index, where the first level as subject identifier (mouse, human, image) and the second level of the index denotes sample repetition identity, in this case - the week during experiment, in which the sample was collected.


```python
display(data_table.iloc[:2,:2].round(3))

```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Fature_1</th>
      <th>Fature_2</th>
    </tr>
    <tr>
      <th>SubjectID</th>
      <th>Week</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">P_10343</th>
      <th>0</th>
      <td>0.001</td>
      <td>0.023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.020</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>


### Shape the data into tensor

We use the `table2tensor` helper function to transform a 2-level (multi)-indexed `pandas.DataFrame` into a 3rd order tensor. 


```python
from mprod import table2tensor
data_tensor, map1, map3 =  table2tensor(data_table)
```

To inspect `table2tensor` operation, we use the resulting *\"mode mappings\"*; `map1` and `map3`  associating each line in the input table to it's coordinates in the resulting tensor.
In the following example, we use the mappings to extract the tensor coordinates corresponding to subject P\_7218's sample from week 52


```python
(data_tensor[map1['P_7218'],:, map3[52]] == data_table.loc[('P_7218',52)].values).all() # True
```

### Applying `TCAM`

```python
from mprod.dimensionality_reduction import TCAM

tca = TCAM()
tca_trans = tca.fit_transform(data_tensor)
```

And that's all there is to it... Really!

Note how similar the code above to what we would have written if we were to apply scikit-lean's `PCA` to the initial tabular data:


```python
from sklearn.decomposition import PCA

pca = PCA()
pca_trans = pca.fit_transform(data_table)
```

The similarity between `TCAM`s interface to that of scikit-learn's `PCA` is not coincidental.
We did our best in order to make `TCAM` as familiar as possible, and allow for high compatibility of `TCAM` with the existing Python ML framework.

### Accessing properties of the transformation


```python
tca_loadings = tca.mode2_loadings  # Obtain TCAM loadings
pca_loadings = pca.components_     # Obtain PCA loadings

tca_var = tca.explained_variance_ratio_*100 # % explained variation per TCA factor
pca_var = pca.explained_variance_ratio_*100 # % explained variation per TCA factor

tca_df = pd.DataFrame(tca_trans)   # Cast TCA scores to dataframe
tca_df.rename(index = dict(map(reversed, map1.items()))
              , inplace = True)    # use the inverse of map1 to denote each row 
                                   # of the TCAM scores with it's subject ID
    
pca_df = pd.DataFrame(pca_trans)   # Cast PCA scores to dataframe
pca_df.index = data_table.index    # anotate PC scores with sample names
```





## References
<a id="1">[1]</a> 
Misha E. Kilmer, Lior Horesh, Haim Avron, and Elizabeth Newman.  Tensor-tensor algebra for optimal representation and compression of multiway data. Proceedings of the National Academy of Sciences, 118(28):e2015851118, jul 2021.

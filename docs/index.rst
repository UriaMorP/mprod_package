.. mprod documentation master file, created by
   sphinx-quickstart on Sun Aug  1 10:11:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

..
   _.. figure:: _static/img/mprod_logo_fav.png


===================================================
:code:`mprod`\: Tensor - Tensor algebraic framework
===================================================

mprod is a software implementation for tensor-tensor algebraic framework derived from the
:math:`\star_{\bf{M}}`-product :footcite:p:`Kilmer`.
The package builds on NumPy\ :footcite:p:`Harris2020` and Scipy\ :footcite:p:`Virtanen2020` libraries to realize
core operations and components required for the algebraic framework.



:mod:`mprod-package` implements the fundamental components required for the :math:`\star_{\mathbf{M}}`-product algebraic
framework; tensor-transpose, tensor-matrix multiplication (domain transforms), face-wise tensor multiplication, and, of
course, the :math:`\star_{\mathbf{M}}` tensor-tensor product (See intro)

In addition, the library offers several basic tensor factorizations such as :mod:`mprod.decompostions.tsvdm`
:footcite:p:`Kilmer` , and :math:`\star_{\mathbf{M}}`-product based dimensionality reduction methods like the
:mod:`mprod.dimensionality_reduction.TCAM` :footcite:p:`mor2021`


.. figure:: _static/img/mprod_tcam_cartoon.png
      :alt: TCAM cartoon
      :class: with-shadow
      :width: 90%
      :align: center

      An introductory cartoon for the TCAM :footcite:p:`mor2021` - an :math:`\star_{\mathbf{M}}`-product based
      dimensionality reduction method for multi-way data.

You can find the software `on github <https://github.com/UriaMorP/mprod_package/>`_.


---------------------------------------------------------

**Installation**
================

Conda install, with the great help of the conda-forge team:

.. code:: bash

    conda install -c conda-forge mprod-package

The conda-forge packages are available for Linux, OS X, and Windows 64 bit. Local testing was done only on Linux.

PyPI install, presuming you have requirements installed (numpy, scipy, pandas, scikit-learn) installed:

.. code:: bash

    pip install mprod-package

-------------------------------------------------------------



Scientific context
------------------

*We live in a multi-dimensional world, immersed in huge volumes of data. This data often involves complex interlinked
structures that span across multiple dimensions. Processes and phenomena also exhibit multi-dimensional behavior,
requiring their models to operate in high dimensional settings*\ .

*Typically, we use matrix algebra to manipulate data, in so-called vector embedded spaces. But such representations
usually don’t take into account the underlying integrity of an object’s dimension, either missing out on high-order
links that go beyond pairwise relations or requiring an overhead in encoding such relations. This is where tensor
algebra comes into play, addressing multiple dimensions*\ .

*But there is a problem. Despite a broad consensus, distilled over centuries of mathematical research, for matrix
algebra, there is no such standard for its multidimensional counterpart, tensor algebra. There have been several
propositions for tensor algebra frameworks over the years* :footcite:p:`Kolda2009`. *Existing techniques that decompose
tensor constructs into simpler tangible entities have limitations and inconsistencies compared to matrix algebra*
:footcite:p:`Hitchcock1927,DeLathauwer2000,Oseledets2011,Tuck1963a`. *These issues have been hindering broad
adoption of tensor algebra into mainstream use*\ .

**The tensor-tensor** :math:`\star_{\bf{M}}`\ **-product framework aims to change that**\ .

*The paper* “**Tensor-Tensor Algebra for Optimal Representationand Compression of Multiway Data**”
:footcite:p:`Kilmer` *describes a way to bridge the gap between matrix and tensor algebra, resulting in new algebraic
constructs that natively represent and manipulate high-dimensional entities, while preserving their multi-order
integrity*\ .

  -- \ **Lior Horesh, IBM research** :footcite:p:`LHoresh`

-------------------------


.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Explore

   examples/examples
   modules/classes

-------------------------

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

----------------------

.. footbibliography::

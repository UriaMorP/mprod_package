.. _tutorials
=========
Tutorials
=========

.. rubric:: Scope and intention
This page presents a collection of tutorial written by the authors of mprod package
and intended to help newcomers in incorporating the machinery offered by the library
in their analysis workflows.

The main (and only) data-scientific tool currently implemented is the TCAM
dimensionality reduction algorithm (REF). We intend to keep expanding the
package content by adding :math:`\star_{\mathbf{M}}`-product based tools
(such as tensor-PLS, tensor-CCA), and we encourage any form of collaboration,
hoping to get good responses, feedback and help from the data-science community.

.. rubric:: Target audience
We, the authors of the library, do not consider ourselves "machine learning experts", and
thus, we don't aim for the expert crowd. Hence, we do not expect the users such expertise.

That said, the library is not - by any means - meant to serve as a **black magic tensor package for dummies**.
Just like with almost everything in machine-learning, using this library for ML related tasks require **some** general
mathematical understanding of ML concepts.
Since the implementation here, is made consistent with `scikit-learn <https://scikit-learn.org/>`_ library to the
maximum possible extent, in order to enable smooth integration within the pythonic ML ecosystem.
For this reason, the users are assumed to know the `scikit-learn <https://scikit-learn.org/>`_ library.
Scikit-learn package offers fantastic documentation, tutorials and examples that are more than enough in order to get
started with machine learning in no time (dear computational biologist, please take your time and at least
try to read it before returning to your search for tools in R).

In addition, deep understanding of the mathematical theory underlying  mprod based tensor algorithms is always a good
idea. A short `primer` section  about the idea of tensor-tensor algebra via the :math:`\star_{\bf{M}}` -product framework
**will soon be added** . For a thorough introduction, we refer the interested readers to the paper
which presented it for the first time :footcite:p:`Kilmer`. Tutorials are grouped by the relevant mprod classes which
implements the different algorithms. For construction and showcase of TCAM refer to :footcite:p:`mor2021`

.. |copy| unicode:: U+2699

--------------------------------


.. _Primer:

------------
⚙ Background
------------
.. toctree::
   :maxdepth: 4

   mprod_primer

.. _TCAM:

----
TCAM
----
.. toctree::
   :maxdepth: 4

   basic_example
   Schirmer2018


.. footbibliography::

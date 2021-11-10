
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
We, the authors of mprod, do not consider ourselves "machine learning experts", and
thus, we don't aim for the expert crowd.

However, mprod, by no means meant to be a **black magic tensor package for dummies**.
As it is the case with almost everything in machine-learning, using this library for machine-learning related tasks
require **some** general mathematical understanding of ML concepts. We set the prerequisites bar on a decent familiarity
of the `scikit-learn <https://scikit-learn.org/>`_ library as a minimal assumption about the background of the package's
intended users. scikit-learn package offers fantastic documentation, tutorials and examples that have more than enough
to get anyone started with machine learning in no time (dear computational biologist, please take your time and at least
try to read it before returning to search tools in R).

In addition, deep understanding of the mathematical theory underlying  mprod based tensor algorithms is always a good
idea. In the following :ref:`primer` section we shortly present and discuss the idea of tensor-tensor algebra via the
:math:`\star_{\bf{M}}` -product framework. For a thorough introduction, we refer the interested readers to the paper
which presented it for the first time :footcite:p:`Kilmer`. Tutorials are grouped by the relevant mprod classes which
implements the different algorithms.



.. toctree::
   :hidden:

   intro


TCAM
----
.. toctree::
   :maxdepth: 2


   Schirmer2018
   basic_example
   supervised_learning


.. footbibliography::
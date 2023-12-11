from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

configuration = {
    "author": 'Uria Mor, Rafael Valdes Mas, Yotam Cohen, Haim Avron',
    "author_email": "uriamo@gmail.com",
    "description": "Software implementation for tensor-tensor m-product framework",
    "long_description_content_type": 'text/markdown',
    "license": "BSD",
    "classifiers": [  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Programming Language :: Python :: 3.10",
        'Programming Language :: Python :: 3 :: Only',
    ],
    "keywords": ["Tensor", "multi way"
        , "omics", "longitudinal"
        , "factorization", "analysis"
        , "TCA", "TCAM", "PCA", "M product"
        , "tensor tensor product"
        , "tSVD", "tSVDM", "tensor decomposition"],
    "name": 'mprod-package',
    "version": '0.0.5a1',
    "packages": find_packages(),
    "long_description": readme,
    "url": "https://github.com/UriaMorP/mprod_package",
    "python_requires": '>=3.6.8',
    "install_requires": [
        "numpy >= 1.19.2",
        "scikit-learn >= 0.24.1",
        "scipy >= 1.5.3",
        "dataclasses >= 0.7; python_version < '3.7'",
        "pandas >= 1.1.5"
    ],
    "extras_require": {
        "dev": ["pytest==6.2.2", ],
        "docs": [
            "sphinx-gallery == 0.9.0",
            "numpydoc == 1.1.0",
            "sphinxcontrib-bibtex == 2.3.0",
            "sphinx-prompt == 1.4.0",
            "nbsphinx == 0.8.6",
            "ipykernel == 5.4.3",
            "seaborn == 0.11.1",
            "jupyter == 1.0.0",
            "myst-parser == 0.15.2",
            "m2r2 == 0.3.1",
            "livereload == 2.6.3",
            "pandoc == 2.0.1",
        ]
    }
}

setup(**configuration)

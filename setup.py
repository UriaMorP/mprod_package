from setuptools import setup

configuration = {
    "name":'mprod-package',
    "version":'0.0.1',
    "packages":['mprod'],
    "long_description":open('README.md').read(),
    "extras_require": {
        "dev":["pytest>=6.2.0",]
    }
}

setup(**configuration)

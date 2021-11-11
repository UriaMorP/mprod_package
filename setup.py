from setuptools import setup

configuration = {
    "name":'mprod_package',
    "version":'0.1dev',
    "packages":['mprod'],
    "long_description":open('README.md').read(),
}

setup(**configuration)

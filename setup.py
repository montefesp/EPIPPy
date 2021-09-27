from __future__ import absolute_import

from setuptools import setup, find_packages
from codecs import open

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='epippy',
    version='0.0.2',
    authors=['Antoine Dubois', 'David Radu'],
    author_email='antoine.dubois@uliege.be',
    description='Expansion Planning Input Preprocessing in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/montefesp/EPIPPy',
    license='MIT',
    package_dir={"": "epippy"},
    packages=find_packages(where='epippy'),
    python_requires='>=3.7',
    install_requires=[
        'pypsa',
        'gdal==2.4.4',
        'pycountry',
        'geopandas',
        'geopy',
        'xlrd',
        'unidecode',
        'dask',
        'xlrd',
        'progressbar2',
        'openpyxl',
        'geokit',
        'glaes',
        'windpowerlib',
        'vresutils'
    ],
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ])

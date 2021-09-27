<a href="https://www.montefiore.uliege.be/"><img src="https://www.montefiore.uliege.be/upload/docs/image/svg-xml/2019-04/montefiore_institute.svg" alt="University of Liège - Montefiore institute" width="230px"></a>

# EPIPPy
Expansion Planning Input Preprocessing in Python

EPIPPy provides a set of functions and scripts to preprocess data that can be then used in an expansion planning model.

## Current capabilities

(Retrieval), preprocessing and management of:
- <a href="https://github.com/montefesp/epippy/tree/master/epippy/generation">generation</a> data (hydro capacities and flows, RES existing capacities, capacity factors, potentials, ...)
- <a href="https://github.com/montefesp/epippy/tree/master/epippy/geographics">geographical</a> data (computation of countries and sub-regions shapes, manipulation of points, codes, ...)
- <a href="https://github.com/montefesp/epippy/tree/master/epippy/indicators/population">population</a> data
- <a href="https://github.com/montefesp/epippy/tree/master/epippy/indicators/emissions">emissions</a> (annual emission computation)
- <a href="https://github.com/montefesp/epippy/tree/master/epippy/load">load</a> (yearly and hourly load computation)
- <a href="https://github.com/montefesp/epippy/tree/master/epippy/technologies">technologies</a> (costs and parameters)
- <a href="https://github.com/montefesp/epippy/tree/master/epippy/topologies">topologies</a>

## Data

For EPIPPy to work properly, it has to have access to a database following a certain structure. This database can be downloaded <a href="ttps://zenodo.org/record/5519081">here</a>.

After downloading the database, the path to the folder containing it must be specified in epippy/\_\_init\_\_.py (to be updated).

## Dependencies
`To be complented`

## Installation

1. Clone git
2. Add epippy to your PYTHONPATH

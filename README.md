# iepy
Input preprocessing for Expansion Planning in Python

iepy provides a set of functions and scripts to preprocess data that can be then used in an expansion planning model.

## Current capabilities

(Retrieval), preprocessing and management of:
- <a href="https://github.com/montefesp/iepy/tree/master/iepy/generation">generation</a> data (hydro capacities and flows, RES existing capacities, capacity factors, potentials, ...)
- <a href="https://github.com/montefesp/iepy/tree/master/iepy/geographics">geographical</a> data (computation of countries and sub-regions shapes, manipulation of points, codes, ...)
- <a href="https://github.com/montefesp/iepy/tree/master/iepy/indicators/population">population</a> data
- <a href="https://github.com/montefesp/iepy/tree/master/iepy/indicators/emissions">emissions</a> (annual emission computation)
- <a href="https://github.com/montefesp/iepy/tree/master/iepy/load">load</a> (yearly and hourly load computation)
- <a href="https://github.com/montefesp/iepy/tree/master/iepy/technologies">technologies</a> (costs and parameters)
- <a href="https://github.com/montefesp/iepy/tree/master/iepy/topologies">topologies</a>

## Data

For iepy to work properly, it has to have access to a database following a certain structure. This database can be downloaded <a href="https://dox.ulg.ac.be/index.php/apps/files/?dir=/py_grid_exp&fileid=268947668">here</a>.

After downloading the database, the path to the folder containing it must be specified in iepy/\_\_init\_\_.py (to be updated).

## Dependencies
`To be complented`

## Installation

1. Clone git
2. Add iepy to your PYTHONPATH

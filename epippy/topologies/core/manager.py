from typing import List, Union

import numpy as np
import pandas as pd

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union
import shapely.wkt

from sklearn.neighbors import NearestNeighbors
from vresutils.graph import voronoi_partition_pts
import scipy as sp
from scipy.sparse import csgraph
import networkx as nx

import pypsa

from epippy.geographics import get_points_in_shape

import logging
logger = logging.getLogger()


def voronoi_special(shape: Union[Polygon, MultiPolygon], centroids: List[List[float]], resolution: float = 0.5):
    """
    Apply a special Voronoi partition of a non-convex polygon based on an approximation of the
    geodesic distance to a set of points which define the centroids of each partition.

    Parameters
    ----------
    shape: Union[Polygon, MultiPolygon]
        Non-convex shape
    centroids: List[List[float]], shape: Nx2
        List of coordinates
    resolution: float (default: 0.5)
        The smaller this value the more precise the geodesic approximation

    Returns
    -------
    List of N Polygons
    """

    # Get all the points in the shape at a certain resolution
    points = get_points_in_shape(shape, resolution)

    # Build a network from these points where each points correspond to a node
    #   and each points is connected to its adjacent points
    adjacency_matrix = np.zeros((len(points), len(points)))
    for i, c_point in enumerate(points):
        adjacency_matrix[i, :] = \
            [1 if np.abs(c_point[0]-point[0]) <= resolution and np.abs(c_point[1]-point[1]) <= resolution else 0
             for point in points]
        adjacency_matrix[i, i] = 0.0
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Find the closest node in the graph corresponding to each centroid
    nbrs = NearestNeighbors(n_neighbors=1).fit(points)
    _, idxs = nbrs.kneighbors(centroids)
    centroids_nodes_indexes = [idx[0] for idx in idxs]

    # For each point, find the closest centroid using shortest path in the graph
    # (i.e approximation of the geodesic distance)
    points_closest_centroid_index = np.zeros((len(points), ))
    points_closest_centroid_length = np.ones((len(points), ))*1000
    for index in centroids_nodes_indexes:
        shortest_paths_length = nx.shortest_path_length(graph, source=index)
        for i in range(len(points)):
            if i in shortest_paths_length and shortest_paths_length[i] < points_closest_centroid_length[i]:
                points_closest_centroid_index[i] = index
                points_closest_centroid_length[i] = shortest_paths_length[i]

    # Compute the classic voronoi partitions of the shape using all points and then join the region
    # corresponding to the same centroid
    voronoi_partitions = voronoi_partition_pts(points, shape)

    return [cascaded_union(voronoi_partitions[points_closest_centroid_index == index])
            for index in centroids_nodes_indexes]


def remove_dangling_branches(branches_df: pd.DataFrame(), buses_ids: Union[List[str], pd.Index]):
    """
    Remove branches that are not connected to any buses.

    Parameters
    ----------
    branches_df: pd.DataFrame
        Dataframe containing the branches (with bus0 and bus1 attributes)
    buses_ids: List[str]
        List of buses ids

    Returns
    -------
    Filtered DataFrame
    """
    return branches_df.loc[branches_df.bus0.isin(buses_ids) & branches_df.bus1.isin(buses_ids)]


def find_closest_links(links_df: pd.DataFrame, new_links_df: pd.DataFrame,
                       distance_upper_bound: float = 1.5) -> pd.Series:
    """
    TODO: complete
    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    This function is originally copied from PyPSA-Eur script base_network.py.
    

    """
    # TODO: this is horribly coded, isn't it?
    #  - first: links and new_links are two dfs that are differently organised...
    #  - second: variable names are not great
    # Get a table where each line contains the end and start of each link
    treecoords = np.asarray([np.asarray(shapely.wkt.loads(s))[[0, -1]].flatten()
                             for s in links_df.geometry])
    # Do the same with the new links but back and forth
    querycoords = np.vstack([new_links_df[['x1', 'y1', 'x2', 'y2']],
                             new_links_df[['x2', 'y2', 'x1', 'y1']]])

    # Built a KDTree for quick nearest-neighbour lookup on original links
    tree = sp.spatial.KDTree(treecoords)
    # For each new link, retrieve distance to nearest neighbor and id of nearest neighbor
    dist, ind = tree.query(querycoords, distance_upper_bound=distance_upper_bound)
    # The algorithm might not find neighbors for some links, it then returns 'number of possible neighbors'
    found_b = ind < len(links_df)
    # Because we added links back and forth, do some magic to retrieve indices of matched new link
    found_i = np.arange(len(new_links_df)*2)[found_b] % len(new_links_df)

    # Create DataFrame containing for each matched new link, the original link to which it matched
    # and the distance between them allowing to remove duplicates
    matched_links_df = pd.DataFrame(dict(D=dist[found_b],
                                         i=links_df.index[ind[found_b] % len(links_df)]),
                                    index=new_links_df.index[found_i])
    matched_links_df = matched_links_df.sort_values(by='D')
    matched_links_df = matched_links_df[lambda ds: ~ds.index.duplicated(keep='first')]
    matched_links_df = matched_links_df.sort_index()

    # Return series matching new links to original ones
    return matched_links_df['i']


def remove_unconnected_components(net: pypsa.Network) -> pypsa.Network:
    """
    TODO: complete
    Parameters
    ----------
    net

    Returns
    -------

    Notes
    -----
    This function is originally copied from PyPSA-Eur script base_network.py.

    """
    _, labels = csgraph.connected_components(net.adjacency_matrix(), directed=False)
    component = pd.Series(labels, index=net.buses.index)

    component_sizes = component.value_counts()
    components_to_remove = component_sizes.iloc[1:]

    logger.info("Removing {} unconnected network components with less than {} buses. In total {} buses."
                .format(len(components_to_remove), components_to_remove.max(), components_to_remove.sum()))

    return net[component == component_sizes.index[0]]

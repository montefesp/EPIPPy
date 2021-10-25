from typing import List, Tuple, Union

import pandas as pd
import numpy as np

from itertools import product
from scipy.spatial import Voronoi

import shapely
import shapely.prepared
from shapely.ops import unary_union, nearest_points
from shapely.errors import TopologicalError
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
import geopy.distance

from epippy.geographics.shapes import get_shapes

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def match_points_to_regions(points: List[Tuple[float, float]], shapes_ds: pd.Series,
                            keep_outside: bool = True, distance_threshold: float = 5.) -> pd.Series:
    """
    Match a set of points to regions by identifying in which region each point falls.

    If keep_outside is True, points that don't fall in any shape but which are close enough to one shape are kept,
    otherwise those points are dumped.

    Parameters
    ----------
    points: List[Tuple[float, float]]
        List of points to assign to regions
    shapes_ds : pd.Series
        Dataframe storing geometries of NUTS regions.
    keep_outside: bool (default: True)
        Whether to keep points that fall outside of shapes
    distance_threshold: float (default: 5.)
        Maximal distance (km) from one shape for points outside of all shapes to be accepted

    Returns
    -------
    points_region_ds : pd.Series
        Series giving for each point the associated region or NA.
    """

    assert len(points) != 0, "Error: List of points is empty."
    assert len(shapes_ds) != 0, "Error Series of shapes is empty."

    points_region_ds = pd.Series(index=pd.MultiIndex.from_tuples(points)).sort_index()
    points = MultiPoint(points)

    # Loop through all shapes and assign points intersect with them
    for index, shape in shapes_ds.items():

        try:
            points_in_region = points.intersection(shape)
        except shapely.errors.TopologicalError:
            logger.warning(f"WARNING: TopologicalError with shape {index}")
            continue

        # After intersection, we can end up into 4 cases:
        #  1 - No points were in the intersection and we obtain an empty geometry
        if points_in_region.is_empty:
            continue
        # 2 - Only one point is in the intersection
        elif isinstance(points_in_region, Point):
            points = points.difference(points_in_region)
            points_in_region = (points_in_region.x, points_in_region.y)
        #  3 - No points were in the intersection and we obtain a MultiPoint of length 0
        elif len(points_in_region) == 0:
            continue
        # 4 - A set of points are in the intersection and we obtained a MultiPoint of length > 0
        else:
            points = points.difference(points_in_region)
            points_in_region = [(point.x, point.y) for point in points_in_region]
        points_region_ds.loc[points_in_region] = index

        # If all points have been assigned to shapes, return
        if points.is_empty:
            return points_region_ds

    # Because of the resolution of the shapes or the accuracy of some points coordinates, some points
    #  might not fall into any shapes but close to their border.
    # If one desires, those points can be added assigned to shapes to which they are close enough.
    if not keep_outside:
        return points_region_ds

    if isinstance(points, Point):
        points = [points]

    # Loop through all points, compute the distance to every shape and add to the closest shape
    #  if the distance is inferior to a maximal distance.
    remaining_points = []
    for point in points:
        closest_points_in_shapes = [nearest_points(point, shapes_ds.loc[subregion])[1] for subregion in shapes_ds.index]
        distances = [geopy.distance.geodesic((point.y, point.x), (c_point.y, c_point.x)).km
                     for c_point in closest_points_in_shapes]
        if min(distances) < distance_threshold:
            closest_index = np.argmin(distances)
            points_region_ds.loc[(point.x, point.y)] = shapes_ds.index.values[closest_index]
        else:
            remaining_points += [point]

    if len(remaining_points) != 0:
        logger.info(f"INFO: These points were not assigned to any shape: "
                    f"{[(point.x, point.y) for point in remaining_points]}.")

    return points_region_ds


def match_points_to_countries(points: List[Tuple[float, float]], countries: List[str]) -> pd.Series:
    """
    Return in which country's region (onshore + offshore) each points falls into.

    Parameters
    ----------
    points: List[Tuple[float, float]]
        List of points defined as tuples (longitude, latitude)
    countries: List[str]
        List of ISO codes of countries

    Returns
    -------
    pd.Series
        Series giving for each point the associated region or NA if the point didn't fall into any region
    """

    shapes = get_shapes(countries, save=True)

    union_shapes = pd.Series(index=countries)
    for country in countries:
        union_shapes[country] = unary_union(shapes.loc[country, 'geometry'])

    # Assign points to each country based on region
    return match_points_to_regions(points, union_shapes)


def get_points_in_shape(shape: Union[Polygon, MultiPolygon], resolution: float,
                        points: List[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
    """
    Return list of coordinates (lon, lat) located inside a geographical shape at a certain resolution.

    Parameters
    ----------
    shape: Polygon or MultiPolygon
        Geographical shape made of points (lon, lat)
    resolution: float
        Longitudinal and latitudinal spatial resolution of points
    points: List[(float, float)] (default: None)
        Points from which to start

    Returns
    -------
    points_in_shape: List[(float, float)]

    """
    if points is None:
        # Generate all points at the given resolution inside a rectangle whose bounds are the
        #  based on the outermost points of the shape.
        minx, miny, maxx, maxy = shape.bounds
        minx = round(minx / resolution) * resolution
        maxx = round(maxx / resolution) * resolution
        miny = round(miny / resolution) * resolution
        maxy = round(maxy / resolution) * resolution
        xs = np.linspace(minx, maxx, num=int((maxx - minx) / resolution) + 1)
        ys = np.linspace(miny, maxy, num=int((maxy - miny) / resolution) + 1)
        points = list(product(xs, ys))
    points = MultiPoint(points)
    points_in_shape = points.intersection(shape)
    if points_in_shape.is_empty:
        points_in_shape = []
    elif isinstance(points_in_shape, Point):
        points_in_shape = [(points_in_shape.x, points_in_shape.y)]
    else:
        points_in_shape = [(point.x, point.y) for point in points_in_shape]

    return points_in_shape


def voronoi_partition(points: List[Tuple[float, float]], shape: Union[Polygon, MultiPolygon],
                      spatial_resolution: float):
    """
    Computes the polygons of a Voronoi partition of a given shape. Function adapted from the vresutils package.

    Attributes
    ----------
    points : List[Tuple[float, float]]
    shape : Union[Polygon, MultiPolygon]
    spatial_resolution : float
        Spatial resolution of the underlying data

    Returns
    -------
    polygons_arr : array of polygon objects
    """

    points = np.asarray(points)

    if len(points) == 1:
        polygons = [shape]
    else:
        xmin, ymin = np.amin(points, axis=0)
        xmax, ymax = np.amax(points, axis=0)
        xspan = max((xmax - xmin), spatial_resolution)
        yspan = max((ymax - ymin), spatial_resolution)

        # to avoid any network positions outside all Voronoi cells, append
        # the corners of a rectangle framing these points
        vor = Voronoi(np.vstack((points,
                                 [[xmin-3.*xspan, ymin-3.*yspan],
                                  [xmin-3.*xspan, ymax+3.*yspan],
                                  [xmax+3.*xspan, ymin-3.*yspan],
                                  [xmax+3.*xspan, ymax+3.*yspan]])))

        polygons = []
        for i in range(len(points)):
            poly = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

            if not poly.is_valid:
                poly = poly.buffer(0)

            poly = poly.intersection(shape)

            polygons.append(poly)

    polygons_arr = np.empty((len(polygons),), 'object')
    polygons_arr[:] = polygons

    return polygons_arr

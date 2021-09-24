from typing import List, Tuple, Union, Dict

import pandas as pd
import numpy as np

from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

from epippy.geographics.points import voronoi_partition
from epippy.geographics import get_points_in_shape
from epippy.geographics.plot import display_polygons
from epippy.technologies import get_config_dict

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def plot_grid_cells(grid_cells_ds: pd.Series, show=False):
    """Plot grid cells (points and regions)"""

    land_50m = cf.NaturalEarthFeature('physical', 'land', '50m',
                                      edgecolor='darkgrey',
                                      facecolor='white')

    axes = []
    for tech in set(grid_cells_ds.index.get_level_values(0)):
        tech_grid_cells_ds = grid_cells_ds.loc[tech]
        ax = display_polygons(tech_grid_cells_ds.values, fill=False, show=False)
        points = list(tech_grid_cells_ds.index)
        xs, ys = zip(*points)
        ax.add_feature(land_50m, linewidth=0.5)
        ax.add_feature(cf.BORDERS.with_scale('50m'), edgecolor='darkgrey', linewidth=0.5)
        ax.scatter(xs, ys, transform=ccrs.PlateCarree(), c='k', zorder=10)
        axes += [ax]

    if show:
        plt.show()
    return axes


def create_grid_cells(shape: Union[Polygon, MultiPolygon], resolution: float) \
        -> (List[Tuple[float, float]], List[Union[Polygon, MultiPolygon]]):
    """Divide a geographical shape by applying voronoi partition."""

    points = get_points_in_shape(shape, resolution)
    if not points:
        return [(shape.centroid.x, shape.centroid.y)], np.array([unary_union(shape)])

    grid_cells = voronoi_partition(points, shape, resolution)
    # Keep only Polygons and MultiPolygons
    for i, shape in enumerate(grid_cells):
        if isinstance(shape, GeometryCollection):
            geos = [geo for geo in shape if isinstance(geo, Polygon) or isinstance(geo, MultiPolygon)]
            grid_cells[i] = unary_union(geos)
    return points, grid_cells


def get_grid_cells(technologies: List[str], resolution: float,
                   onshore_shape: pd.Series = None,
                   offshore_shape: pd.Series = None) -> pd.Series:
    """
    Divide shapes in grid cell for a list of technologies.

    Parameters
    ----------
    technologies: List[str]
        List of technologies for which we want to generate grid cells.
    resolution: float
        Spatial resolution at which the grid cells must be defined.
    onshore_shape: pd.Series (default: None)
        Onshore geographical scope.
    offshore_shape: pd.Series (default: None)
        Offshore geographical scope.

    Returns
    -------
    pd.Series
        Series indicating for each technology and each grid cell defined for this technology the associated
        grid cell shape.

    """

    assert len(technologies) != 0, 'Error: Empty list of technologies.'

    # Determine if tech are onshore- or offshore-based
    tech_config = get_config_dict(technologies, ["onshore"])

    # Check the right shapes have been passed
    for tech in technologies:
        is_onshore = tech_config[tech]["onshore"]
        shape = onshore_shape if is_onshore else offshore_shape
        assert shape is not None, f"Error: Missing {'onshore' if is_onshore else 'offshore'} " \
                                  f"shapes for technology {tech}"

    # Divide onshore and offshore shapes at a given resolution
    onshore_points, onshore_grid_cells_shapes = [], np.array([])
    if onshore_shape is not None:
        for r in onshore_shape.index:
            union_sh = unary_union(onshore_shape.loc[r])
            onshore_points_region, onshore_grid_cells_shapes_region = create_grid_cells(union_sh, resolution)
            if not onshore_points_region:
                logger.warning(f"No points at given resolution falls into {r} onshore. "
                               "Taking the centroid of the shape and the shape itself.")
            onshore_points.extend(onshore_points_region)
            onshore_grid_cells_shapes = np.append(onshore_grid_cells_shapes, onshore_grid_cells_shapes_region)

    offshore_points, offshore_grid_cells_shapes = [], np.array([])
    if offshore_shape is not None:
        for r in offshore_shape.index:
            union_sh = unary_union(offshore_shape.loc[r])
            offshore_points_region, offshore_grid_cells_shapes_region = create_grid_cells(union_sh, resolution)
            if not offshore_points_region:
                logger.warning(f"No points at given resolution falls into {r} offshore. "
                               "Taking the centroid of the shape and the shape itself.")
            offshore_points.extend(offshore_points_region)
            offshore_grid_cells_shapes = np.append(offshore_grid_cells_shapes, offshore_grid_cells_shapes_region)

    # Collect onshore and offshore grid cells for each technology
    tech_point_tuples = []
    grid_cells_shapes = np.array([])
    for i, tech in enumerate(technologies):
        is_onshore = tech_config[tech]["onshore"]
        points = onshore_points if is_onshore else offshore_points
        tech_grid_cell_shapes = onshore_grid_cells_shapes if is_onshore else offshore_grid_cells_shapes
        grid_cells_shapes = np.append(grid_cells_shapes, tech_grid_cell_shapes)
        tech_point_tuples += [(tech, point[0], point[1]) for point in points]

    grid_cells = pd.Series(grid_cells_shapes, index=pd.MultiIndex.from_tuples(tech_point_tuples))
    grid_cells.index.names = ["Technology Name", "Longitude", "Latitude"]

    return grid_cells.sort_index()


if __name__ == "__main__":

    from epippy.geographics.shapes import get_shapes
    from epippy.geographics.codes import get_subregions
    from epippy.geographics.plot import display_polygons
    from shapely.ops import cascaded_union

    region = get_subregions('EU_BK')
    shapes = get_shapes(region, which='offshore', save=False)

    # display_polygons(shapes.geometry.values, fill=True, show=True)
    # points, grid_cells_ds = create_grid_cells(shapes_union, resolution=1.0)
    grid_cells_ds = get_grid_cells(['wind_offshore'], 0.5, onshore_shape=None, offshore_shape=shapes.geometry.dropna())

    plot_grid_cells(grid_cells_ds, show=True)

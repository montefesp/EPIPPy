"""
DISCLAIMER:
The functions in this script are greatly inspired by code from PyPSA-Eur
and a lot of code directly comes from there.
"""
from typing import List

import numpy as np
import pandas as pd
import geopy.distance

import pypsa

from iepy.geographics import get_shapes, match_points_to_regions

import logging
logger = logging.getLogger()


def simplify_network_to_380(net: pypsa.Network) -> (pypsa.Network, pd.Series):
    """
    TODO: complete

    Parameters
    ----------
    net

    Returns
    -------

    """

    logger.info("Mapping all network lines onto a single 380kV layer")

    net.buses['v_nom'] = 380.

    # Reset num_parallel, v_nom, type and s_nom for non 380kV lines
    linetype_380, = net.lines.loc[net.lines.v_nom == 380., 'type'].unique()
    non380_lines_b = net.lines.v_nom != 380.
    net.lines.loc[non380_lines_b, 'num_parallel'] *= (net.lines.loc[non380_lines_b, 'v_nom'] / 380.)**2
    net.lines.loc[non380_lines_b, 'v_nom'] = 380.
    net.lines.loc[non380_lines_b, 'type'] = linetype_380
    net.lines.loc[non380_lines_b, 's_nom'] = (
        np.sqrt(3) * net.lines['type'].map(net.line_types.i_nom) *
        net.lines.bus0.map(net.buses.v_nom) * net.lines.num_parallel
    )

    # Remove all transformers and merge buses at different voltage levels
    # Create a series associating the starting bus of each transformer to its end bus
    trafo_map = pd.Series(net.transformers.bus1.values, index=net.transformers.bus0.values)
    # Remove duplicate elements that have the same starting bus
    trafo_map = trafo_map[~trafo_map.index.duplicated(keep='first')]
    # Update ending bus it two transfos follow each other
    several_trafo_b = trafo_map.isin(trafo_map.index)
    trafo_map.loc[several_trafo_b] = trafo_map.loc[several_trafo_b].map(trafo_map)
    # Find buses without transfos starting on them and a transfo starting and finishing at those buses
    missing_buses_i = net.buses.index.difference(trafo_map.index)
    trafo_map = trafo_map.append(pd.Series(missing_buses_i, missing_buses_i))

    # Set containing {'Load', 'Generator', 'Store', 'StorageUnit', ShuntImpedance', 'Link', 'Line', 'Transformer'}
    # Update bus information in all components DataFrame
    for c in net.one_port_components | net.branch_components:
        df = net.df(c)
        for col in df.columns:
            if col.startswith('bus'):
                df[col] = df[col].map(trafo_map)

    # Remove all transformers
    net.mremove("Transformer", net.transformers.index)
    net.mremove("Bus", net.buses.index.difference(trafo_map))

    return net, trafo_map


def cluster_network(net: pypsa.Network, nuts_codes: List[str]):

    # Get shapes of regions (onshore and offshore)
    shapes = get_shapes(nuts_codes, which='onshore')

    # --- Buses --- #
    # TODO: this is shit --> should return a list keeping the index of the original points
    def add_region(lon, lat):
        try:
            region_code = matched_locs[lon, lat]
            # Need the if because some points are exactly at the same position
            return region_code if (isinstance(region_code, str) or isinstance(region_code, float)
                                   or isinstance(region_code, int)) else region_code.iloc[0]
        except (AttributeError, KeyError):
            return None
    # plants_region_ds.loc[pp_df_in_country.index] = \
    #     pp_df_in_country[["lon", "lat"]].apply(lambda x: add_region(x[0], x[1]), axis=1)

    buses_positions = net.buses[['x', 'y']].apply(lambda p: (p.x, p.y), axis=1).tolist()
    matched_locs = match_points_to_regions(buses_positions, shapes["geometry"], keep_outside=False).dropna()
    old_to_new_buses_map = net.buses[["x", "y"]].apply(lambda p: add_region(p.x, p.y), axis=1).dropna()
    nuts_codes_with_buses = list(set(old_to_new_buses_map))

    # Merge buses to centroid of countries (even offshore ones)
    buses_df = pd.DataFrame(columns=["bus_id", "x", "y"])
    buses_df["bus_id"] = nuts_codes_with_buses
    buses_df = buses_df.set_index('bus_id')
    regions = shapes.loc[nuts_codes_with_buses, 'geometry']
    buses_df['onshore_region'] = regions
    buses_df["x"] = regions.centroid.apply(lambda p: p.x)
    buses_df["y"] = regions.centroid.apply(lambda p: p.y)

    print(buses_df)

    def compute_distance(bus0, bus1):
        bus0_x, bus0_y = buses_df.loc[bus0, ['x', 'y']]
        bus1_x, bus1_y = buses_df.loc[bus1, ['x', 'y']]
        return geopy.distance.geodesic((bus0_y, bus0_x), (bus1_y, bus1_x)).km

    # --- Lines --- #
    # Remove lines associated to buses that have been removed
    net.lines = net.lines.loc[net.lines.bus0.isin(old_to_new_buses_map.index)
                              & net.lines.bus1.isin(old_to_new_buses_map.index)]
    # Assign new bus to lines
    net.lines.bus0 = net.lines.bus0.map(old_to_new_buses_map)
    net.lines.bus1 = net.lines.bus1.map(old_to_new_buses_map)

    # Remove lines that have the same starting and end bus
    net.lines = net.lines[net.lines.bus0 != net.lines.bus1]

    # Merge lines with same starting and end bus
    # Get all lines connected to the same bus in the same direction
    net.lines[['bus0', 'bus1']] = [sorted((bus0, bus1)) for (bus0, bus1) in net.lines[['bus0', 'bus1']].values]
    # Get pairs of connected bus
    connected_buses = set([(bus0, bus1) for (bus0, bus1) in net.lines[['bus0', 'bus1']].values.tolist()])
    all_lines_df = pd.DataFrame()
    # Compute reactance of lines
    net.lines["x"] = net.lines["length"]*net.lines['type'].map(net.line_types.x_per_length)
    for bus0, bus1 in connected_buses:
        # line_index = f"{bus0}-{bus1}"
        # lines_df.loc[line_index, ['bus0', 'bus1']] = (bus0, bus1)

        # Get all lines in original network that were connected to bus0 and bus1
        old_lines_df = net.lines[(net.lines.bus0 == bus0) & (net.lines.bus1 == bus1)]

        # Merge those lines by voltage level

        # TODO: Parameters to consider
        #  - name: ok
        #  - bus0: ok
        #  - bus1: ok
        #  - underground: ? --> affect cost
        #  - under_construction: already dealt with before
        #  - tags: nope
        #  - geometry: nope (replaced by straight line)
        #  - length: how? -> just take fly-distance (times 1.25 like in pypsa-eur?)
        #  - x: how?
        #  - v_nom: how?
        #  - num_parallel: how?
        #    Use sth similar to this: net.lines.loc[non380_lines_b, 'num_parallel'] *= (net.lines.loc[non380_lines_b, 'v_nom'] / 380.)**2 ?
        #  - s_nom: how?

        lines_df = old_lines_df.groupby("type")['v_nom'].unique().apply(lambda x: x[0]).to_frame()
        lines_df['s_nom'] = old_lines_df.groupby("type")['s_nom'].sum()
        lines_df['num_parallel'] = old_lines_df.groupby("type")['num_parallel'].sum()
        lines_df['x'] = old_lines_df.groupby("type")['x'].apply(lambda x: 1/sum(1/x))
        lines_df["line_id"] = lines_df['v_nom'].apply(lambda x: f"{bus0}-{bus1}-{x}")
        lines_df["bus0"] = bus0
        lines_df["bus1"] = bus1
        lines_df["length"] = lines_df[['bus0', 'bus1']].apply(lambda x: compute_distance(x.bus0, x.bus1), axis=1)
        lines_df = lines_df.reset_index().set_index('line_id')

        all_lines_df = pd.concat((all_lines_df, lines_df))

    print(all_lines_df)

    # --- Links --- #
    # Remove lines associated to buses that have been removed
    net.links = net.links.loc[net.links.bus0.isin(old_to_new_buses_map.index)
                              & net.links.bus1.isin(old_to_new_buses_map.index)]
    net.links.bus0 = net.links.bus0.map(old_to_new_buses_map)
    net.links.bus1 = net.links.bus1.map(old_to_new_buses_map)
    # Remove links that have the same starting and end bus
    net.links = net.links[net.links.bus0 != net.links.bus1]
    # Merge links with same starting and end bus
    # Get all links connected to the same bus in the same direction
    net.links[['bus0', 'bus1']] = [sorted((str(bus0), str(bus1))) for (bus0, bus1) in net.links[['bus0', 'bus1']].values]
    # Get pairs of connected bus
    connected_buses = set([(bus0, bus1) for (bus0, bus1) in net.links[['bus0', 'bus1']].values.tolist()])
    links_df = pd.DataFrame(index=[f"{bus0}-{bus1}" for (bus0, bus1) in connected_buses],
                            columns=['bus0', 'bus1', 'p_nom', 'length'])
    for bus0, bus1 in connected_buses:
        link_index = f"{bus0}-{bus1}"
        links_df.loc[link_index, ['bus0', 'bus1']] = (bus0, bus1)
        # Get all links in original network that were connected to bus0 and bus1
        old_links_df = net.links[(net.links.bus0 == bus0) & (net.links.bus1 == bus1)]
        # Add capacities
        links_df.loc[link_index, 'p_nom'] = old_links_df['p_nom'].sum()
    links_df["length"] = links_df[['bus0', 'bus1']].apply(lambda x: compute_distance(x.bus0, x.bus1), axis=1)

    # TODO: Parameters to consider
    #  - name: ok
    #  - bus0: ok
    #  - bus1: ok
    #  - carrier: nope --> all dc
    #  - underground: how? --> affect cost
    #  - underwater_fraction: how? --> affect cost
    #  - under_construction: already dealt with before
    #  - tags: nope
    #  - geometry: nope (replaced by straight line)
    #  - length: how? -> just take fly-distance (times 1.25 like in pypsa-eur?)
    #  - p_nom: how?
    #  - num_parallel: how?

    # TODO: What about transformers?
    #

    print(net.links)
    print(links_df)

    clustered_net = pypsa.Network()
    clustered_net.import_components_from_dataframe(buses_df, 'Bus')
    clustered_net.import_components_from_dataframe(all_lines_df, 'Line')
    clustered_net.import_components_from_dataframe(links_df, 'Link')

    print(clustered_net.buses.onshore_region)

    return clustered_net

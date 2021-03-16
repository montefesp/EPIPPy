"""
DISCLAIMER:
The functions in this script are greatly inspired by code from PyPSA-Eur
and a lot of code directly comes from there.
"""
from typing import List, Dict
import yaml

import numpy as np
import pandas as pd
from six import iteritems
from itertools import product

from shapely.ops import unary_union
from shapely.geometry import Point, LineString
import shapely.wkt
import scipy as sp
import geopandas as gpd
import networkx as nx

import pypsa

from iepy.geographics import get_shapes
from iepy.topologies.core.manager import remove_dangling_branches, find_closest_links, remove_unconnected_components
from iepy import data_path

import logging
logger = logging.getLogger()


def get_oid(df):
    if "tags" in df.columns:
        return df.tags.str.extract('"oid"=>"(\d+)"', expand=False)
    else:
        return pd.Series(np.nan, df.index)


def get_country(df):
    if "tags" in df.columns:
        return df.tags.str.extract('"country"=>"([A-Z]{2})"', expand=False)
    else:
        return pd.Series(np.nan, df.index)


def load_buses_from_eg(countries: List[str], voltages: List[float]):
    """
    TODO: complete

    Parameters
    ----------
    countries

    Returns
    -------

    """

    buses_fn = f"{data_path}/topologies/pypsa_entsoe_gridkit/source/buses.csv"
    buses_df = pd.read_csv(buses_fn, quotechar="'",
                        true_values='t', false_values='f',
                        dtype=dict(bus_id="str"))
    buses_df = buses_df.set_index("bus_id").drop(['station_id'], axis=1).rename(columns=dict(voltage='v_nom'))

    # Replace dc column by carrier column
    buses_df['carrier'] = buses_df.pop('dc').map({True: 'DC', False: 'AC'})
    buses_df['under_construction'] = buses_df['under_construction'].fillna(False).astype(bool)

    # remove all buses outside of all countries including exclusive economic zones (offshore)
    countries_shapes_ds = get_shapes(countries)["geometry"]
    # TODO: what does prep do?
    # europe_shape_prepped = shapely.prepared.prep(europe_shape)
    all_countries_shape = unary_union(countries_shapes_ds)
    buses_in_europe_b = buses_df[['x', 'y']].apply(lambda p: all_countries_shape.contains(Point(p)), axis=1)

    # Remove all buses which are not at the desired voltage
    buses_with_v_nom_to_keep_b = buses_df.v_nom.isin(voltages) | buses_df.v_nom.isnull()
    logger.info("Removing buses with voltages {}"
                "".format(pd.Index(buses_df.v_nom.unique()).dropna().difference(voltages)))

    return buses_df.loc[buses_in_europe_b & buses_with_v_nom_to_keep_b]


def load_links_from_eg(buses_df: pd.DataFrame, links_config: Dict, countries):
    """
    TODO: complete

    Parameters
    ----------
    buses_ids: List[str]


    Returns
    -------

    """

    links_fn = f"{data_path}/topologies/pypsa_entsoe_gridkit/source/links.csv"
    links_df = pd.read_csv(links_fn, quotechar="'", true_values='t', false_values='f',
                           dtype=dict(link_id='str', bus0='str', bus1='str', under_construction="bool"))
    links_df = links_df.set_index('link_id')
    links_df['length'] /= 1e3
    # Add DC line parameters
    links_df['carrier'] = 'DC'
    # hotfix
    links_df.loc[links_df.bus1 == '6271', 'bus1'] = '6273'

    # Remove links not connected to any desired buses
    links_df= remove_dangling_branches(links_df, buses_df.index)

    # Add TYNDP links if required
    if links_config.get('include_tyndp'):
        buses_df, links_df = add_links_from_tyndp(buses_df, links_df, countries)

    # Check whether we have any links left
    if links_df.empty:
        return links_df

    # Set p_max_pu
    p_max_pu = links_config.get('p_max_pu', 1.)
    links_df['p_max_pu'] = p_max_pu
    links_df['p_min_pu'] = -p_max_pu

    # Add p_nom
    links_df = add_links_p_nom(links_df)

    return buses_df, links_df


def add_links_from_tyndp(buses_df, links_df, countries):
    """

    TODO: complete

    # Read links in links_tyndp.csv, remove links whose buses are not in Europe,
    # check overlap between the two datasets, add non-overlapping links

    Parameters
    ----------
    buses_df
    links_df

    Returns
    -------

    """
    links_tyndp_fn = f"{data_path}/topologies/pypsa_entsoe_gridkit/source/links_tyndp.csv"
    links_tyndp_df = pd.read_csv(links_tyndp_fn)

    # Remove all links from list which lie outside all of the desired countries
    countries_shapes_ds = get_shapes(countries)["geometry"]
    # TODO: what does prep do?
    # europe_shape_prepped = shapely.prepared.prep(europe_shape)
    all_countries_shape = unary_union(countries_shapes_ds)
    x1y1_in_europe_b = links_tyndp_df[['x1', 'y1']].apply(lambda p: all_countries_shape.contains(Point(p)), axis=1)
    x2y2_in_europe_b = links_tyndp_df[['x2', 'y2']].apply(lambda p: all_countries_shape.contains(Point(p)), axis=1)
    is_within_covered_countries_b = x1y1_in_europe_b & x2y2_in_europe_b

    # If not all links are in the desired area, update links_df accordingly
    if not is_within_covered_countries_b.all():
        logger.info("TYNDP links outside of the covered area (skipping): " +
                    ", ".join(links_tyndp_df.loc[~ is_within_covered_countries_b, "Name"]))

        links_tyndp_df = links_tyndp_df.loc[is_within_covered_countries_b]
        # If none of the links are in the desired area, return the original dataframes
        if links_tyndp_df.empty:
            return buses_df, links_df

    # Some TYNDP were associated to specific links in advance
    # TODO:
    #   - How are those association made?
    #   - All this is a bit obscure
    has_replaces_b = links_tyndp_df.replaces.notnull()
    oids = dict(Bus=get_oid(buses_df), Link=get_oid(links_df))
    keep_b = dict(Bus=pd.Series(True, index=buses_df.index),
                  Link=pd.Series(True, index=links_df.index))
    # For each of those links, get the oids of the link(s) it replaces
    for reps in links_tyndp_df.loc[has_replaces_b, 'replaces']:
        for comps in reps.split(':'):
            oids_to_remove = comps.split('.')
            c = oids_to_remove.pop(0)
            keep_b[c] &= ~oids[c].isin(oids_to_remove)
    buses_df = buses_df.loc[keep_b['Bus']]
    links_df = links_df.loc[keep_b['Link']]

    # Find correspondence between the two datasets (i.e. gridkit and tyndp) to see if some TYNDP links are
    # already present
    #  - 0.2 corresponds approximately to 20km tolerances
    #  - 'j' will contain an id of links in gridkit for TYNDP link that were matched
    links_tyndp_df["j"] = find_closest_links(links_df, links_tyndp_df, distance_upper_bound=0.20)
    if links_tyndp_df["j"].notnull().any():
        logger.info("TYNDP links already in the dataset (skipping): " +
                    ", ".join(links_tyndp_df.loc[links_tyndp_df["j"].notnull(), "Name"]))
        # Keep only links that were not already in Gridkit
        links_tyndp_df = links_tyndp_df.loc[links_tyndp_df["j"].isnull()]
        if links_tyndp_df.empty:
            return buses_df, links_df

    # Find to which bus remaining TYNDP links should be connected
    tree = sp.spatial.KDTree(buses_df[['x', 'y']])
    _, ind0 = tree.query(links_tyndp_df[["x1", "y1"]])
    ind0_b = ind0 < len(buses_df)
    links_tyndp_df.loc[ind0_b, "bus0"] = buses_df.index[ind0[ind0_b]]

    _, ind1 = tree.query(links_tyndp_df[["x2", "y2"]])
    ind1_b = ind1 < len(buses_df)
    links_tyndp_df.loc[ind1_b, "bus1"] = buses_df.index[ind1[ind1_b]]

    # Check if some links were not connected to any buses
    links_tyndp_located_b = links_tyndp_df["bus0"].notnull() & links_tyndp_df["bus1"].notnull()
    if not links_tyndp_located_b.all():
        logger.warning("Did not find connected buses for TYNDP links (skipping): " + ", ".join(links_tyndp_df.loc[~links_tyndp_located_b, "Name"]))
        links_tyndp_df = links_tyndp_df.loc[links_tyndp_located_b]

    logger.info("Adding the following TYNDP links: " + ", ".join(links_tyndp_df["Name"]))

    links_tyndp_df = links_tyndp_df[["bus0", "bus1"]].assign(
        carrier='DC',
        p_nom=links_tyndp_df["Power (MW)"],
        length=links_tyndp_df["Length (given) (km)"].fillna(links_tyndp_df["Length (distance*1.2) (km)"]),
        under_construction=True,
        underground=False,
        geometry=(links_tyndp_df[["x1", "y1", "x2", "y2"]]
                  .apply(lambda s: str(LineString([[s.x1, s.y1], [s.x2, s.y2]])), axis=1)),
        tags=('"name"=>"' + links_tyndp_df["Name"] + '", ' +
              '"ref"=>"' + links_tyndp_df["Ref"] + '", ' +
              '"status"=>"' + links_tyndp_df["status"] + '"')
    )

    links_tyndp_df.index = "T" + links_tyndp_df.index.astype(str)

    return buses_df, links_df.append(links_tyndp_df, sort=True)


def add_links_p_nom(links_df: pd.DataFrame):
    """

    TODO: complete

    # According to the documentation:
    # https://pypsa-eur.readthedocs.io/en/latest/preparation/prepare_links_p_nom.html#links
    # This file is generated by the script prepare_links_p_nom which extract capacities of HVDC links from Wikipedia
    # So the links in links.csv do not have capacities associated to them (those from tyndp do, at least for some)
    # so get it from this Wikipedia file...

    Parameters
    ----------
    links_df

    Returns
    -------

    """
    # Add p_nom from some Wikipedia source because this information is not contained in the gridkit dataset
    # TODO: specify how this file is generate
    links_p_nom_fn = f"{data_path}/topologies/pypsa_entsoe_gridkit/source/links_p_nom.csv"
    links_p_nom_df = pd.read_csv(links_p_nom_fn)

    # Filter links that are not in operation anymore
    removed_b = links_p_nom_df.Remarks.str.contains('Shut down|Replaced', na=False)
    links_p_nom_df = links_p_nom_df[~removed_b]

    # Find closest link for all links in links_p_nom
    links_p_nom_df['j'] = find_closest_links(links_df, links_p_nom_df)

    links_p_nom_df = links_p_nom_df.groupby(['j'], as_index=False).agg({'Power (MW)': 'sum'})

    # TODO: change this name?
    p_nom_df = links_p_nom_df.dropna(subset=["j"]).set_index("j")["Power (MW)"]

    # Don't update p_nom if it's already set
    p_nom_unset = p_nom_df.drop(links_df.index[links_df.p_nom.notnull()], errors='ignore') \
        if "p_nom" in links_df else p_nom_df
    links_df.loc[p_nom_unset.index, "p_nom"] = p_nom_unset

    return links_df


# TODO: could maybe merge this into load_links_from_eg
def set_links_underwater_fraction(net: pypsa.Network, countries: List[str]):
    """
    TODO: complete

    Set what portion of the link is under water --> has an influence on the cost computed in script add_electricity

    Parameters
    ----------
    net
    countries

    Returns
    -------

    """
    if net.links.empty: return

    if not hasattr(net.links, 'geometry'):
        net.links['underwater_fraction'] = 0.
    else:
        # TODO: are there links associated to some countries going through other countries eez?
        offshore_shape = unary_union(get_shapes(countries, which='offshore')["geometry"])
        links = gpd.GeoSeries(net.links.geometry.dropna().map(shapely.wkt.loads))
        net.links['underwater_fraction'] = links.intersection(offshore_shape).length / links.length


def load_converters_from_eg(buses_ids: List[str], links_config: Dict):
    """
    TODO: complete
    Parameters
    ----------
    buses_ids

    Returns
    -------

    """
    converters_fn = f"{data_path}/topologies/pypsa_entsoe_gridkit/source/converters.csv"
    converters_df = pd.read_csv(converters_fn, quotechar="'",
                                true_values='t', false_values='f',
                                dtype=dict(converter_id='str', bus0='str', bus1='str'))
    converters_df = converters_df.set_index('converter_id')
    converters_df['carrier'] = 'B2B'

    # Remove unconnected converters
    converters_df = remove_dangling_branches(converters_df, buses_ids)

    # Add electrical parameters
    p_max_pu = links_config.get('p_max_pu', 1.)
    converters_df['p_max_pu'] = p_max_pu
    converters_df['p_min_pu'] = -p_max_pu

    # TODO: why?
    converters_df['p_nom'] = 2000

    # Converters are combined with links
    converters_df['under_construction'] = False
    converters_df['underground'] = False

    return converters_df


def replace_b2b_converter_at_country_border_by_link(net: pypsa.Network):
    # Affects only the B2B converter in Lithuania at the Polish border at the moment
    # TODO: check thate ther is a country associated to each bus --> associated with set_countries_and_substations
    buscntry = net.buses.country
    linkcntry = net.links.bus0.map(buscntry)
    converters_i = net.links.index[(net.links.carrier == 'B2B') & (linkcntry == net.links.bus1.map(buscntry))]

    def findforeignbus(G, i):
        cntry = linkcntry.at[i]
        for busattr in ('bus0', 'bus1'):
            b0 = net.links.at[i, busattr]
            for b1 in G[b0]:
                if buscntry[b1] != cntry:
                    return busattr, b0, b1
        return None, None, None

    for i in converters_i:
        G = net.graph()
        busattr, b0, b1 = findforeignbus(G, i)
        if busattr is not None:
            comp, line = next(iter(G[b0][b1]))
            if comp != "Line":
                logger.warning("Unable to replace B2B `{}` expected a Line, but found a {}"
                            .format(i, comp))
                continue

            net.links.at[i, busattr] = b1
            net.links.at[i, 'p_nom'] = min(net.links.at[i, 'p_nom'], net.lines.at[line, 's_nom'])
            net.links.at[i, 'carrier'] = 'DC'
            net.links.at[i, 'underwater_fraction'] = 0.
            net.links.at[i, 'length'] = net.lines.at[line, 'length']

            net.remove("Line", line)
            net.remove("Bus", b0)

            logger.info("Replacing B2B converter `{}` together with bus `{}` and line `{}` by an HVDC tie-line {}-{}"
                        .format(i, b0, line, linkcntry.at[i], buscntry.at[b1]))


def load_lines_from_eg(buses_ids: List[str], lines_config: Dict, v_noms: List[float],
                       line_types_df: pd.DataFrame):
    """
    TODO: complete
    Parameters
    ----------
    buses_ids

    Returns
    -------

    """
    lines_fn = f"{data_path}/topologies/pypsa_entsoe_gridkit/source/lines.csv"
    lines_df = pd.read_csv(lines_fn, quotechar="'", true_values='t', false_values='f',
                           dtype=dict(line_id='str', bus0='str', bus1='str',
                                      underground="bool", under_construction="bool"))
    lines_df = lines_df.set_index('line_id').rename(columns=dict(voltage='v_nom', circuits='num_parallel'))
    lines_df['length'] /= 1e3

    # Remove unconnected lines
    lines_df = remove_dangling_branches(lines_df, buses_ids)

    # Set electrical parameters
    for v_nom in v_noms:
        lines_df.loc[lines_df["v_nom"] == v_nom, 'type'] = lines_config['types'][v_nom]
    lines_df['s_max_pu'] = lines_config['s_max_pu']

    # Set lines s_nom (i.e. limit of apparent power in MVA) from line types
    lines_df['s_nom'] = (
        np.sqrt(3) * lines_df['type'].map(line_types_df.i_nom) *
        lines_df['v_nom'] * lines_df.num_parallel
    )

    return lines_df


# TODO: could this be merged with load_lines_from_eg and load_links_from_eg
def adjust_capacities_of_under_construction_branches(net: pypsa.Network, config_lines: Dict, config_links: Dict)\
        -> pypsa.Network:
    """
    TODO: complete

    Allows to set under construction links and lines to 0 or remove them completely,
    + remove some unconnected components that might appear as a result

    Parameters
    ----------
    net
    config_lines
    config_links

    Returns
    -------

    """
    lines_mode = config_lines.get('under_construction', 'undef')
    if lines_mode == 'zero':
        net.lines.loc[net.lines.under_construction, 'num_parallel'] = 0.
        net.lines.loc[net.lines.under_construction, 's_nom'] = 0.
    elif lines_mode == 'remove':
        net.mremove("Line", net.lines.index[net.lines.under_construction])
    elif lines_mode != 'keep':
        logger.warning("Unrecognized configuration for `lines: under_construction` = `{}`. Keeping under construction lines.")

    links_mode = config_links.get('under_construction', 'undef')
    if links_mode == 'zero':
        net.links.loc[net.links.under_construction, "p_nom"] = 0.
    elif links_mode == 'remove':
        net.mremove("Link", net.links.index[net.links.under_construction])
    elif links_mode != 'keep':
        logger.warning("Unrecognized configuration for `links: under_construction` = `{}`. Keeping under construction links.")

    if lines_mode == 'remove' or links_mode == 'remove':
        # We might need to remove further unconnected components
        net = remove_unconnected_components(net)

    return net


def load_transformers_from_eg(buses_ids: List[str], transformers_config: Dict):
    """
    TODO: complete
    Parameters
    ----------
    buses_ids

    Returns
    -------

    """
    transformers_fn = f"{data_path}/topologies/pypsa_entsoe_gridkit/source/transformers.csv"
    transformers_df = pd.read_csv(transformers_fn, quotechar="'",
                                  true_values='t', false_values='f',
                                  dtype=dict(transformer_id='str', bus0='str', bus1='str'))
    transformers_df = transformers_df.set_index('transformer_id')

    # Remove unconnected transformers
    transformers_df = remove_dangling_branches(transformers_df, buses_ids)

    # Add transformer parameters
    transformers_df["x"] = transformers_config.get('x', 0.1)
    transformers_df["s_nom"] = transformers_config.get('s_nom', 2000)
    transformers_df['type'] = transformers_config.get('type', '')

    return transformers_df


def apply_parameter_corrections(net: pypsa.Network):
    """
    TODO: complete

    Parameters
    ----------
    net

    Returns
    -------

    """
    # TODO: how is this file computed?
    corrections_fn = f"{data_path}/topologies/pypsa_entsoe_gridkit/source/parameter_corrections.yaml"
    with open(corrections_fn) as f:
        corrections = yaml.safe_load(f)

    if corrections is None:
        return

    for component, attrs in iteritems(corrections):
        df = net.df(component)
        oid = get_oid(df)
        if attrs is None: continue

        for attr, repls in iteritems(attrs):
            for i, r in iteritems(repls):
                if i == 'oid':
                    r = oid.map(repls["oid"]).dropna()
                elif i == 'index':
                    r = pd.Series(repls["index"])
                else:
                    raise NotImplementedError()
                inds = r.index.intersection(df.index)
                df.loc[inds, attr] = r[inds].astype(df[attr].dtype)


# TODO: what are they doing with substations?
def set_countries_and_substations(net: pypsa.Network, countries: List[str]):

    buses = net.buses

    def buses_in_shape(shape):
        shape = shapely.prepared.prep(shape)
        return pd.Series(
            np.fromiter((shape.contains(Point(x, y))
                        for x, y in buses.loc[:,["x", "y"]].values),
                        dtype=bool, count=len(buses)),
            index=buses.index
        )

    shapes = get_shapes(countries)
    # country_shapes = gpd.read_file(snakemake.input.country_shapes).set_index('name')['geometry']
    country_shapes = shapes[~shapes.offshore]['geometry']
    # offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes).set_index('name')['geometry']
    offshore_shapes = shapes[shapes.offshore]['geometry']
    substation_b = buses['symbol'].str.contains('substation|converter station', case=False)

    def prefer_voltage(x, which):
        index = x.index
        if len(index) == 1:
            return pd.Series(index, index)
        key = (x.index[0]
               if x['v_nom'].isnull().all()
               else getattr(x['v_nom'], 'idx' + which)())
        return pd.Series(key, index)

    gb = buses.loc[substation_b].groupby(['x', 'y'], as_index=False,
                                         group_keys=False, sort=False)
    bus_map_low = gb.apply(prefer_voltage, 'min')
    lv_b = (bus_map_low == bus_map_low.index).reindex(buses.index, fill_value=False)
    bus_map_high = gb.apply(prefer_voltage, 'max')
    hv_b = (bus_map_high == bus_map_high.index).reindex(buses.index, fill_value=False)

    onshore_b = pd.Series(False, buses.index)
    offshore_b = pd.Series(False, buses.index)

    for country in countries:
        onshore_shape = country_shapes[country]
        onshore_country_b = buses_in_shape(onshore_shape)
        onshore_b |= onshore_country_b

        buses.loc[onshore_country_b, 'country'] = country

        if country not in offshore_shapes.index: continue
        offshore_country_b = buses_in_shape(offshore_shapes[country])
        offshore_b |= offshore_country_b

        buses.loc[offshore_country_b, 'country'] = country

    # Only accept buses as low-voltage substations (where load is attached), if
    # they have at least one connection which is not under_construction
    has_connections_b = pd.Series(False, index=buses.index)
    for b, df in product(('bus0', 'bus1'), (net.lines, net.links)):
        has_connections_b |= ~ df.groupby(b).under_construction.min()

    buses['substation_lv'] = lv_b & onshore_b & (~ buses['under_construction']) & has_connections_b
    buses['substation_off'] = (offshore_b | (hv_b & onshore_b)) & (~ buses['under_construction'])

    c_nan_b = buses.country.isnull()
    if c_nan_b.sum() > 0:
        c_tag = get_country(buses.loc[c_nan_b])
        c_tag.loc[~c_tag.isin(countries)] = np.nan
        net.buses.loc[c_nan_b, 'country'] = c_tag

        c_tag_nan_b = net.buses.country.isnull()

        # Nearest country in path length defines country of still homeless buses
        # Work-around until commit 705119 lands in pypsa release
        net.transformers['length'] = 0.
        graph = net.graph(weight='length')
        net.transformers.drop('length', axis=1, inplace=True)

        for b in net.buses.index[c_tag_nan_b]:
            df = (pd.DataFrame(dict(pathlength=nx.single_source_dijkstra_path_length(graph, b, cutoff=200)))
                  .join(net.buses.country).dropna())
            assert not df.empty, "No buses with defined country within 200km of bus `{}`".format(b)
            net.buses.at[b, 'country'] = df.loc[df.pathlength.idxmin(), 'country']

        logger.warning("{} buses are not in any country or offshore shape,"
                       " {} have been assigned from the tag of the entsoe map,"
                       " the rest from the next bus in terms of pathlength."
                       .format(c_nan_b.sum(), c_nan_b.sum() - c_tag_nan_b.sum()))

    return buses

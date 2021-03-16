"""
DISCLAIMER:
The functions in this script are greatly inspired by code from PyPSA-Eur
and a lot of code directly comes from there.

Description
-----------
The rule :mod:`simplify_network` does up to four things:

1. Create an equivalent transmission network in which all voltage levels are mapped to the 380 kV level by the function
``simplify_network(...)``.

2. DC only sub-networks that are connected at only two buses to the AC network are reduced to a single representative
link in the function ``simplify_links(...)``. The components attached to buses in between are moved to the nearest
 endpoint. The grid connection cost of offshore wind generators are added to the captial costs of the generator.

3. Stub lines and links, i.e. dead-ends of the network, are sequentially removed from the network in the function
 ``remove_stubs(...)``. Components are moved along.

4. Optionally, if an integer were provided for the wildcard ``{simpl}`` (e.g. ``networks/elec_s500.nc``),
 the network is clustered to this number of clusters with the routines from the ``cluster_network`` rule with
 the function ``cluster_network.cluster(...)``. This step is usually skipped!

"""
from typing import Dict

import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse.csgraph import connected_components, dijkstra

from six import iteritems

import pypsa
from pypsa.io import import_components_from_dataframe, import_series_from_dataframe
from pypsa.networkclustering import busmap_by_stubs, aggregategenerators, aggregateoneport

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
    print(net.lines.loc[net.lines.v_nom == 380., 'type'].unique())
    linetype_380, = net.lines.loc[net.lines.v_nom == 380., 'type'].unique()
    non380_lines_b = net.lines.v_nom != 380.
    net.lines.loc[non380_lines_b, 'num_parallel'] *= (net.lines.loc[non380_lines_b, 'v_nom'] / 380.)**2
    net.lines.loc[non380_lines_b, 'v_nom'] = 380.
    net.lines.loc[non380_lines_b, 'type'] = linetype_380
    net.lines.loc[non380_lines_b, 's_nom'] = (
        np.sqrt(3) * net.lines['type'].map(net.line_types.i_nom) *
        net.lines.bus0.map(net.buses.v_nom) * net.lines.num_parallel
    )

    # Replace transformers by lines
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
        print(c)
        df = net.df(c)
        print(df)
        for col in df.columns:
            if col.startswith('bus'):
                df[col] = df[col].map(trafo_map)

    # Remove all transformers
    net.mremove("Transformer", net.transformers.index)
    net.mremove("Bus", net.buses.index.difference(trafo_map))

    return net, trafo_map


def _prepare_connection_costs_per_link(n):
    if n.links.empty:
        return {}

    costs = load_costs(n.snapshot_weightings.sum() / 8760, snakemake.input.tech_costs,
                       snakemake.config['costs'], snakemake.config['electricity'])

    connection_costs_per_link = {}

    for tech in snakemake.config['renewable']:
        if tech.startswith('offwind'):
            connection_costs_per_link[tech] = (
                n.links.length * snakemake.config['lines']['length_factor'] *
                (n.links.underwater_fraction * costs.at[tech + '-connection-submarine', 'capital_cost'] +
                 (1. - n.links.underwater_fraction) * costs.at[tech + '-connection-underground', 'capital_cost'])
            )

    return connection_costs_per_link


def _compute_connection_costs_to_bus(n, busmap, connection_costs_per_link=None, buses=None):
    if connection_costs_per_link is None:
        connection_costs_per_link = _prepare_connection_costs_per_link(n)

    if buses is None:
        buses = busmap.index[busmap.index != busmap.values]

    connection_costs_to_bus = pd.DataFrame(index=buses)

    for tech in connection_costs_per_link:
        adj = n.adjacency_matrix(weights=pd.concat(dict(Link=connection_costs_per_link[tech].reindex(n.links.index),
                                                        Line=pd.Series(0., n.lines.index))))

        costs_between_buses = dijkstra(adj, directed=False, indices=n.buses.index.get_indexer(buses))
        connection_costs_to_bus[tech] = costs_between_buses[np.arange(len(buses)),
                                                            n.buses.index.get_indexer(busmap.loc[buses])]

    return connection_costs_to_bus


def _adjust_capital_costs_using_connection_costs(n, connection_costs_to_bus):
    for tech in connection_costs_to_bus:
        tech_b = n.generators.carrier == tech
        costs = n.generators.loc[tech_b, "bus"].map(connection_costs_to_bus[tech]).loc[lambda s: s>0]
        if not costs.empty:
            n.generators.loc[costs.index, "capital_cost"] += costs
            logger.info("Displacing {} generator(s) and adding connection costs to capital_costs: {} "
                        .format(tech, ", ".join("{:.0f} Eur/MW/a for `{}`".format(d, b) for b, d in costs.iteritems())))


def _aggregate_and_move_components(net, busmap, connection_costs_to_bus, aggregate_one_ports={"Load", "StorageUnit"}):
    def replace_components(net, c, df, pnl):
        net.mremove(c, net.df(c).index)

        import_components_from_dataframe(net, df, c)
        for attr, df in iteritems(pnl):
            if not df.empty:
                import_series_from_dataframe(net, df, c, attr)

    _adjust_capital_costs_using_connection_costs(net, connection_costs_to_bus)

    generators, generators_pnl = aggregategenerators(net, busmap)
    replace_components(net, "Generator", generators, generators_pnl)

    for one_port in aggregate_one_ports:
        df, pnl = aggregateoneport(net, busmap, component=one_port)
        replace_components(net, one_port, df, pnl)

    buses_to_del = net.buses.index.difference(busmap)
    net.mremove("Bus", buses_to_del)
    for c in net.branch_components:
        df = net.df(c)
        net.mremove(c, df.index[df.bus0.isin(buses_to_del) | df.bus1.isin(buses_to_del)])


def simplify_links(net: pypsa.Network, links_config: Dict) -> (pypsa.Network, pd.Series):
    """
    TODO: complete
    Parameters
    ----------
    net

    Returns
    -------

    """
    # Complex multi-node links are folded into end-points
    logger.info("Simplifying connected link components")

    if net.links.empty:
        return net, net.buses.index.to_series()

    # Determine connected link components, ignore all links but DC
    adjacency_matrix = net.adjacency_matrix(branch_components=['Link'],
                                            weights=dict(Link=(net.links.carrier == 'DC').astype(float)))

    _, labels = connected_components(adjacency_matrix, directed=False)
    labels = pd.Series(labels, net.buses.index)

    G = net.graph()

    def split_links(nodes):
        nodes = frozenset(nodes)

        seen = set()
        supernodes = {m for m in nodes
                      if len(G.adj[m]) > 2 or (set(G.adj[m]) - nodes)}

        for u in supernodes:
            for m, ls in iteritems(G.adj[u]):
                if m not in nodes or m in seen: continue

                buses = [u, m]
                links = [list(ls)] #[name for name in ls]]

                while m not in (supernodes | seen):
                    seen.add(m)
                    for m2, ls in iteritems(G.adj[m]):
                        if m2 in seen or m2 == u: continue
                        buses.append(m2)
                        links.append(list(ls)) # [name for name in ls])
                        break
                    else:
                        # stub
                        break
                    m = m2
                if m != u:
                    yield pd.Index((u, m)), buses, links
            seen.add(u)

    busmap = net.buses.index.to_series()

    connection_costs_per_link = _prepare_connection_costs_per_link(net)
    connection_costs_to_bus = pd.DataFrame(0., index=net.buses.index, columns=list(connection_costs_per_link))

    for lbl in labels.value_counts().loc[lambda s: s > 2].index:

        for b, buses, links in split_links(labels.index[labels == lbl]):
            if len(buses) <= 2: continue

            logger.debug('nodes = {}'.format(labels.index[labels == lbl]))
            logger.debug('b = {}\nbuses = {}\nlinks = {}'.format(b, buses, links))

            m = sp.spatial.distance_matrix(net.buses.loc[b, ['x', 'y']],
                                           net.buses.loc[buses[1:-1], ['x', 'y']])
            busmap.loc[buses] = b[np.r_[0, m.argmin(axis=0), 1]]
            connection_costs_to_bus.loc[buses] += _compute_connection_costs_to_bus(net, busmap, connection_costs_per_link, buses)

            all_links = [i for _, i in sum(links, [])]

            p_max_pu = links_config.get('p_max_pu', 1.)
            lengths = net.links.loc[all_links, 'length']
            name = lengths.idxmax() + '+{}'.format(len(links) - 1)
            params = dict(
                carrier='DC',
                bus0=b[0], bus1=b[1],
                length=sum(net.links.loc[[i for _, i in l], 'length'].mean() for l in links),
                p_nom=min(net.links.loc[[i for _, i in l], 'p_nom'].sum() for l in links),
                underwater_fraction=sum(lengths/lengths.sum() * net.links.loc[all_links, 'underwater_fraction']),
                p_max_pu=p_max_pu,
                p_min_pu=-p_max_pu,
                underground=False,
                under_construction=False
            )

            logger.info("Joining the links {} connecting the buses {} to simple link {}"
                        .format(", ".join(all_links), ", ".join(buses), name))

            net.mremove("Link", all_links)

            static_attrs = net.components["Link"]["attrs"].loc[lambda df: df.static]
            for attr, default in static_attrs.default.iteritems(): params.setdefault(attr, default)
            net.links.loc[name] = pd.Series(params)

            # n.add("Link", **params)

    logger.debug("Collecting all components using the busmap")

    _aggregate_and_move_components(net, busmap, connection_costs_to_bus)
    return net, busmap


def remove_stubs(net: pypsa.Network):
    logger.info("Removing stubs")

    busmap = busmap_by_stubs(net)  # ['country'])

    connection_costs_to_bus = _compute_connection_costs_to_bus(net, busmap)

    _aggregate_and_move_components(net, busmap, connection_costs_to_bus)

    return net, busmap
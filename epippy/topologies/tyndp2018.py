from typing import List
from os.path import isfile, isdir
from os import makedirs

import pandas as pd

import pypsa

import matplotlib.pyplot as plt
import shapely.wkt
from shapely.geometry import Polygon
import geopy.distance

from epippy.geographics import get_shapes
from epippy.technologies import get_costs

from epippy import data_path


def preprocess(plotting=True) -> None:
    """
    Pre-process tyndp-country buses and links information.

    Parameters
    ----------
    plotting: bool
        Whether to plot the results
    """

    generated_dir = f"{data_path}topologies/tyndp2018/generated/"
    if not isdir(generated_dir):
        makedirs(generated_dir)

    # Create links
    link_data_fn = f"{data_path}topologies/tyndp2018/source/Input Data.xlsx"
    # Read TYNDP2018 (NTC 2027, reference grid) data
    #     - ST (Sustainable Transition): targets reached by national regulation, emission trading schemes and subsidies,
    #                                    maximising the use of existing infrastructure
    #     - DG (Distributed Generation): prosumers at the centre - small-scale generation, batteries and fuel-switching
    #                                    society engaged and empowered
    #     - GCA (Global Climate Action): full-speed global decarbonisation, large-scale renewables
    links = pd.read_excel(link_data_fn, sheet_name="NTC", index_col=0, skiprows=[0, 2],
                          usecols=[0, 3, 4, 5, 6, 7, 8, 9, 10],
                          names=["link", "in", "out", "st_in", "st_out", "dg_in", "dg_out", "gca_in", "gca_out"])

    # Get NTC as the minimum capacity between the two flow directions.
    links["p_nom"] = links[["in", "out"]].max(axis=1)
    links["p_nom_st"] = links[["st_in", "st_out"]].max(axis=1)
    links["p_nom_dg"] = links[["dg_in", "dg_out"]].max(axis=1)
    links["p_nom_gca"] = links[["gca_in", "gca_out"]].max(axis=1)
    links["bus0"] = links.index.str[:2]
    links["bus1"] = [i[1][:2] for i in links.index.str.split('-')]

    # Remove links which do not cross international borders.
    links_crossborder = links[links["bus0"] != links["bus1"]].copy()
    links_crossborder["id"] = links_crossborder["bus0"] + '-' + links_crossborder["bus1"]
    # Sum all capacities belonging to the same border and convert from MW to GW.
    links = links_crossborder.groupby("id")[["p_nom", "p_nom_st", "p_nom_dg", "p_nom_gca"]].sum() / 1000.

    links["id"] = links.index.values
    links["bus0"] = links["id"].apply(lambda k: k.split('-')[0])
    links["bus1"] = links["id"].apply(lambda k: k.split('-')[1])

    # A subset of links are assumed to be DC connections.
    dc_set = {'BE-GB', 'CY-GR', 'DE-GB', 'DE-NO', 'DE-SE', 'DK-GB', 'DK-NL', 'DK-NO', 'DK-PL', 'DK-SE',
              'EE-FI', 'ES-FR', 'FR-GB', 'FR-IE', 'GB-IE', 'GB-IS', 'GB-NL', 'GB-NO', 'GR-IT', 'GR-TR',
              'IT-ME', 'IT-MT', 'IT-TN', 'LT-SE', 'PL-SE', 'NL-NO'}
    links["carrier"] = links["id"].apply(lambda x: 'DC' if x in dc_set else 'AC')
    # A connection between Rep. of Ireland (IE) and Northern Ireland (NI) is considered in the TYNDP, yet as NI is the
    # TODO: this is north ireland --> need to add it to the capacity between IE-GB
    # ISO2 code of Nicaragua, this results in weird results. Thus, the connection is dropped, as IE-GB links exist.
    links = links[~links.index.str.contains("NI")]

    # Create buses
    buses_names = []
    for name in links.index:
        buses_names += name.split("-")
    buses_names = sorted(list(set(buses_names)))
    # buses = pd.DataFrame(index=buses_names, columns=["x", "y", "country", "region", "onshore"])
    buses = pd.DataFrame(index=buses_names, columns=["x", "y", "country", "onshore_region", "offshore_region"])
    buses.index.names = ["id"]
    buses.country = list(buses.index)
    # buses.onshore = True

    # Get shape of each country
    # buses.region = get_shapes(buses.index.values, which='onshore', save=True)["geometry"]
    shapes = get_shapes(buses.index.values, save=True)
    # Crop regions going too far north
    nordics = ["FI", "NO", "SE"]
    intersection_poly = Polygon([(0., 50.), (0., 66.5), (40., 66.5), (40., 50.)])
    shapes.loc[nordics, "geometry"] = shapes.loc[nordics, "geometry"].apply(lambda x: x.intersection(intersection_poly))
    # Add regions to buses
    buses.onshore_region = shapes[~shapes.offshore]["geometry"]
    offshore_shapes = shapes[shapes.offshore]["geometry"]
    buses.loc[offshore_shapes.index, "offshore_region"] = offshore_shapes

    centroids = [region.centroid for region in buses.onshore_region]
    buses.x = [c.x for c in centroids]
    buses.y = [c.y for c in centroids]

    for item in buses.index:
        if item == 'NO':
            buses.loc[item, 'x'] = 10.2513
            buses.loc[item, 'y'] = 60.2416
        elif item == 'SE':
            buses.loc[item, 'x'] = 15.2138
            buses.loc[item, 'y'] = 59.3386
        elif item == 'DK':
            buses.loc[item, 'x'] = 9.0227
            buses.loc[item, 'y'] = 56.1997
        elif item == 'GB':
            buses.loc[item, 'x'] = -1.2816
            buses.loc[item, 'y'] = 52.7108
        elif item == 'HR':
            buses.loc[item, 'x'] = 15.89
            buses.loc[item, 'y'] = 45.7366
        elif item == 'GR':
            buses.loc[item, 'x'] = 21.57
            buses.loc[item, 'y'] = 40.19
        elif item == 'FI':
            buses.loc[item, 'x'] = 24.82
            buses.loc[item, 'y'] = 61.06

    # Adding length to the links
    def compute_distance(bus0, bus1):
        bus0_x, bus0_y = buses.loc[bus0, ['x', 'y']]
        bus1_x, bus1_y = buses.loc[bus1, ['x', 'y']]
        return geopy.distance.geodesic((bus0_y, bus0_x), (bus1_y, bus1_x)).km
    links["length"] = links[['bus0', 'bus1']].apply(lambda x: compute_distance(x.bus0, x.bus1), axis=1)

    if plotting:
        from epippy.topologies.core.plot import plot_topology
        plot_topology(buses, links)
        plt.show()

    # buses.region = buses.region.astype(str)
    buses.onshore_region = buses.onshore_region.astype(str)
    buses.offshore_region = buses.offshore_region.astype(str)
    buses.to_csv(f"{generated_dir}buses.csv")
    links.to_csv(f"{generated_dir}links.csv")


def get_topology(network: pypsa.Network, countries: List[str] = None,
                 p_nom_extendable: bool = True, extension_multiplier: float = None, extension_base: str = 'GCA',
                 use_ex_line_cap: bool = True, p_max_pu: float = 1.0,
                 plot: bool = False) -> pypsa.Network:
    """
    Load the e-highway network topology (buses and links) using PyPSA.

    Parameters
    ----------
    network: pypsa.Network
        Network instance
    countries: List[str] (default: None)
        List of ISO codes of countries for which we want the tyndp topology.
    p_nom_extendable: bool (default: True)
        Whether line capacity is allowed to be expanded
    extension_multiplier: float (default: None)
        By how much the capacity can be extended if extendable. If None, no limit on expansion.
    extension_base: str (default: GCA)
        TYNDP 2040 scenario to use as the basis for computing the max potential of links, can be one of:
        - ST (Sustainable Transition): targets reached by national regulation, emission trading schemes and subsidies,
                                       maximising the use of existing infrastructure
                                       ~ 180 GW and 94 TWkm
        - DG (Distributed Generation): prosumers at the centre - small-scale generation, batteries and fuel-switching
                                       society engaged and empowered
                                       ~ 190 GW and 99 TWkm
        - GCA (Global Climate Action): full-speed global decarbonisation, large-scale renewables
                                       ~ 200 GW and 103 TWkm
        The three scenarios are quite similar in terms of NTCs with GCA being the most generous.
    use_ex_line_cap: bool (default True)
        Whether to use existing line capacity
    p_max_pu: float (default: 1.0)
        Maximal dispatch per unit of p_nom
    plot: bool (default: False)
        Whether to show loaded topology or not

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    assert countries is None or len(countries) != 0, "Error: Countries list must not be empty. If you want to " \
                                                     "obtain, the full topology, don't pass anything as argument."
    assert extension_base in ['ST', 'DG', 'GCA'], f"Error: extension_base must be one of ST, DG or GCA, " \
                                                     f"received {extension_base}."

    topology_dir = f"{data_path}topologies/tyndp2018/generated/"
    buses_fn = f"{topology_dir}buses.csv"
    assert isfile(buses_fn), f"Error: Buses are undefined. Please run 'preprocess'."
    buses = pd.read_csv(buses_fn, index_col='id')
    links_fn = f"{topology_dir}links.csv"
    assert isfile(links_fn), f"Error: Links are undefined. Please run 'preprocess'."
    links = pd.read_csv(links_fn, index_col='id')

    if countries is not None:
        # Check if there is a bus for each country considered
        missing_countries = set(countries) - set(buses.index)
        assert not missing_countries, f"Error: No buses exist for the following countries: {missing_countries}"
        # Remove buses that are not associated with the considered countries
        buses = buses.loc[buses.index.isin(countries)]
    countries = buses.index

    # Converting polygons strings to Polygon object
    for region_type in ["onshore_region", "offshore_region"]:
        regions = buses[region_type].values
        # Convert strings
        for i, region in enumerate(regions):
            if isinstance(region, str):
                regions[i] = shapely.wkt.loads(region)

    # If we have only one bus, add it to the network and return
    if len(buses) == 1:
        network.import_components_from_dataframe(buses, "Bus")
        return network

    # Remove links for which one of the two end buses has been removed
    links = pd.DataFrame(links.loc[links.bus0.isin(buses.index) & links.bus1.isin(buses.index)])

    # Removing offshore buses that are not connected anymore
    connected_buses = sorted(list(set(links["bus0"]).union(set(links["bus1"]))))
    buses = buses.loc[connected_buses]

    disconnected_onshore_bus = set(countries) - set(buses.index)
    assert not disconnected_onshore_bus, f"Error: Buses {disconnected_onshore_bus} were disconnected."

    if not use_ex_line_cap:
        links['p_nom'] = 0
    links['p_nom_min'] = links['p_nom']
    links['p_max_pu'] = p_max_pu
    links['p_min_pu'] = -p_max_pu  # Making the link bi-directional
    links['p_nom_extendable'] = p_nom_extendable
    if p_nom_extendable:
        # Choose p_nom_max based on some TYNDP 2040 scenario
        if extension_base == 'ST':
            links['p_nom_max'] = links['p_nom_st']
        elif extension_base == 'DG':
            links['p_nom_max'] = links['p_nom_dg']
        else:
            links['p_nom_max'] = links['p_nom_gca']
        links = links.drop(['p_nom_st', 'p_nom_dg', 'p_nom_gca'], axis=1)
        if extension_multiplier is not None:
            links['p_nom_max'] = (links['p_nom_max']*extension_multiplier).round(3)
            links['p_nom_max'] = links[['p_nom_max', 'p_nom_min']].max(axis=1)
        else:
            links['p_nom_max'] = "inf"
    links['capital_cost'] = pd.Series(index=links.index)
    for idx in links.index:
        carrier = links.loc[idx].carrier
        cap_cost, _ = get_costs(carrier, sum(network.snapshot_weightings['objective']))
        links.loc[idx, ('capital_cost', )] = cap_cost * links.length.loc[idx]

    network.import_components_from_dataframe(buses, "Bus")
    network.import_components_from_dataframe(links, "Link")

    if plot:
        from epippy.topologies.core.plot import plot_topology
        plot_topology(buses, links)
        plt.show()

    return network


if __name__ == "__main__":
    # preprocess(True)

    if 1:
        from pypsa import Network
        from epippy.geographics import get_subregions
        import matplotlib.pyplot as plt

        topology_dir = f"{data_path}topologies/tyndp2018/generated/"
        links_fn = f"{topology_dir}links.csv"
        links_ = pd.read_csv(links_fn, index_col='id')
        diff_st_dg = (links_.p_nom_dg - links_.p_nom_st).abs()
        diff_st_gca = (links_.p_nom_gca - links_.p_nom_st).abs()
        diff_dg_gca = (links_.p_nom_gca - links_.p_nom_dg).abs()
        pd.concat((diff_st_gca, diff_st_dg, diff_dg_gca), axis=1).max(axis=1).plot(kind='bar')
        plt.figure()
        links_.p_nom_st.plot(ls='-', marker='.', c='b', alpha=0.5)
        links_.p_nom_dg.plot(ls='-', marker='.', c='r', alpha=0.5)
        links_.p_nom_gca.plot(ls='-', marker='.', c='g', alpha=0.5)
        plt.xticks(ticks=range(len(links_)), labels=list(links_.index), rotation='90')
        plt.grid()
        plt.show()

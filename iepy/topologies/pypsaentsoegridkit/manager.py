"""This script is greatly inspired by code from PyPSA-Eur and a lot of code directly comes from there."""

from pypsa import Network


from iepy.topologies.pypsaentsoegridkit.base_network import *
from iepy.topologies.pypsaentsoegridkit.cluster_network import *


def set_electrical_parameters_lines(net: pypsa.Network, lines_config: Dict, v_noms: List[float]):
    # Set electrical parameters
    print(v_noms)
    for v_nom in v_noms:
        net.lines.loc[net.lines["v_nom"] == v_nom, 'type'] = lines_config['types'][v_nom]
    net.lines['s_max_pu'] = lines_config['s_max_pu']

    # Set lines s_nom (i.e. limit of apparent power in MVA) from line types
    net.lines['s_nom'] = (
        np.sqrt(3) * net.lines['type'].map(net.line_types.i_nom) *
        net.lines['v_nom'] * net.lines.num_parallel
    )


def set_electrical_parameters_links(net: pypsa.Network, links_config: Dict):
    # Set p_max_pu
    p_max_pu = links_config.get('p_max_pu', 1.)
    net.links['p_max_pu'] = p_max_pu
    net.links['p_min_pu'] = -p_max_pu


def set_electrical_parameters_transformers(net: pypsa.Network, transformers_config: Dict):
    # Add transformer parameters
    net.transformers['x'] = transformers_config.get('x', 0.1)
    net.transformers['s_nom'] = transformers_config.get('s_nom', 2000)
    net.transformers['type'] = transformers_config.get('type', '')


# def preprocess(net: Network, config: Dict, countries: List[str], voltages: List[float], plot: bool = False):
def preprocess(plot: bool = False):
    # TODO:
    #  - figure out what we really need
    #    - converters?
    #    - transformers?
    #  - figure out if lines and/or links can be extended
    #    - see prepare_network
    #  - Probably best to do all these steps once in a preprocess function and then just
    #   remove unwanted components at run time

    # Load main components
    # buses_df = load_buses_from_eg(countries, voltages)
    buses_df = load_buses_from_eg()
    # buses_df, links_df = load_links_from_eg(buses_df, config['links'], countries)
    buses_df, links_df = load_links_from_eg(buses_df)
    # converters_df = load_converters_from_eg(buses_df.index, config["links"])
    converters_df = load_converters_from_eg(buses_df.index)
    # lines_df = load_lines_from_eg(buses_df.index, config["lines"], voltages, net.line_types)
    lines_df = load_lines_from_eg(buses_df.index)
    # transformers_df = load_transformers_from_eg(buses_df.index, config["transformers"])
    transformers_df = load_transformers_from_eg(buses_df.index)

    # Add everything to the network
    net = Network()
    net.import_components_from_dataframe(buses_df, "Bus")
    net.import_components_from_dataframe(lines_df, "Line")
    net.import_components_from_dataframe(transformers_df, "Transformer")
    net.import_components_from_dataframe(links_df, "Link")
    net.import_components_from_dataframe(converters_df, "Link")

    # Update a bunch of parameters for given components according to parameters_correction.yaml
    apply_parameter_corrections(net)
    # Remove subnetworks with less than a given number of components
    net = remove_unconnected_components(net)
    # Determine to which country each bus (onshore or offshore) belong and do some stuff with substations
    # set_countries_and_substations(net, countries)
    set_countries_and_substations(net)
    # Set what portion of the link is under water
    # set_links_underwater_fraction(net, countries)
    set_links_underwater_fraction(net)
    # I have no clue what this is for...
    replace_b2b_converter_at_country_border_by_link(net)

    # Save base network
    net.export_to_csv_folder(f"{data_path}topologies/pypsa_entsoe_gridkit/generated/base_network/")

    if plot:
        import matplotlib.pyplot as plt
        net.plot(bus_sizes=0.001)
        plt.show()

    return net


def load_topology(nuts_codes, config, voltages: List[float] = None, plot: bool = False):

    net = Network()
    net.import_from_csv_folder(f"{data_path}topologies/pypsa_entsoe_gridkit/generated/base_network/")

    if 1:
        import matplotlib.pyplot as plt
        net.plot(bus_sizes=0.001)
        plt.show()

    exit()

    # Remove all buses outside desired regions
    region_shapes_ds = get_shapes(nuts_codes, save=True)["geometry"]
    buses_in_nuts_regions = \
        net.buses[['x', 'y']].apply(lambda p: any([shape.contains(Point(p)) for shape in region_shapes_ds]), axis=1)
    net.buses = net.buses[buses_in_nuts_regions]

    if 0:
        plt.figure()
        net.plot(bus_sizes=0.001)

    # Remove all buses which are not at the desired voltage
    buses_with_v_nom_to_keep_b = net.buses.v_nom.isnull()
    if voltages is not None:
        buses_with_v_nom_to_keep_b |= net.buses.v_nom.isin(voltages)
        logger.info("Removing buses with voltages {}"
                    "".format(pd.Index(net.buses.v_nom.unique()).dropna().difference(voltages)))
    net.buses = net.buses[buses_with_v_nom_to_keep_b]

    if 1:
        plt.figure()
        net.plot(bus_sizes=0.001)
        plt.show()

    # Remove dangling branches
    net.lines = remove_dangling_branches(net.lines, net.buses.index)
    net.links = remove_dangling_branches(net.links, net.buses.index)
    net.transformers = remove_dangling_branches(net.transformers, net.buses.index)

    # Set electrical parameters
    set_electrical_parameters_lines(net, config['lines'], net.buses.v_nom.dropna().unique().tolist())
    set_electrical_parameters_links(net, config['links'])
    set_electrical_parameters_transformers(net, config['transformers'])

    # Allows to set under construction links and lines to 0 or remove them completely,
    # and remove some unconnected components that might appear as a result
    net = adjust_capacities_of_under_construction_branches(net, config['lines'], config['links'])

    # Remove unconnected components

    # TODO: allow to simplify the network (i.e. convert everything to 380) or not ?

    net = cluster_network(net, nuts_codes)

    if plot:
        from iepy.topologies.core.plot import plot_topology
        all_lines = pd.concat((net.links[['bus0', 'bus1']], net.lines[['bus0', 'bus1']]))
        plot_topology(net.buses, all_lines)
        plt.show()

    return net


if __name__ == '__main__':
    config_ = {"lines": {'s_max_pu': 0.7,
                         'types': {132.: "Al/St 240/40 2-bundle 220.0",
                                   220.: "Al/St 240/40 2-bundle 220.0",
                                   300.: "Al/St 240/40 3-bundle 300.0",
                                   380.: "Al/St 240/40 4-bundle 380.0"}},
               "links": {'p_max_pu': 1.0,
                         'include_tyndp': True},
               "transformers": {}}

    voltages_ = [220., 300., 380.]
    from iepy.geographics import get_subregions, get_nuts_codes, revert_iso2_codes
    # TODO: does not work with bosnia and kosovo
    countries_ = get_subregions("EU2")
    # countries_ = ['MK']
    nuts_codes_ = get_nuts_codes(2, 2016, revert_iso2_codes(countries_))
    # Some weird BEZ, LUZ, etc...
    nuts_codes_ = [code for code in nuts_codes_ if not code.endswith('Z')]

    if 0:
        net_ = preprocess(True)

    else:
        # net_ = simplify_network(net_, config_)
        load_topology(nuts_codes_, config_, voltages_, True)

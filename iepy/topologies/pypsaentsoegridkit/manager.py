"""This script is greatly inspired by code from PyPSA-Eur and a lot of code directly comes from there."""

import pandas as pd

from pypsa import Network

from iepy import data_path

from iepy.topologies.pypsaentsoegridkit.base_network import *
from iepy.topologies.pypsaentsoegridkit.simplify_network import *


def load_topology(net: Network, config: Dict, countries: List[str], voltages: List[float]):
    # TODO:
    #  - figure out what we really need
    #    - converters?
    #    - transformers?
    #  - figure out if lines and/or links can be extended
    #    - see prepare_network
    #  - Probably best to do all these steps once in a preprocess function and then just
    #   remove unwanted components at run time

    # Load main components
    buses_df = load_buses_from_eg(countries, voltages)
    buses_df, links_df = load_links_from_eg(buses_df, config['links'], countries)
    converters_df = load_converters_from_eg(buses_df.index, config["links"])
    lines_df = load_lines_from_eg(buses_df.index, config["lines"], voltages, net.line_types)
    transformers_df = load_transformers_from_eg(buses_df.index, config["transformers"])

    # Add everything to the network
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
    set_countries_and_substations(net, countries)
    # Set what portion of the link is under water
    set_links_underwater_fraction(net, countries)
    # I have no clue what this is for...
    replace_b2b_converter_at_country_border_by_link(net)
    # Allows to set under construction links and lines to 0 or remove them completely,
    # and remove some unconnected components that might appear as a result
    net = adjust_capacities_of_under_construction_branches(net, config['lines'], config['links'])

    return net


def simplify_network(net: Network, config: Dict):

    # TODO:
    #  - see simplify_network --> transforms the transmission grid to a 380kV only equivalent network
    net, trafo_map = simplify_network_to_380(net)

    net, simplify_links_map = simplify_links(net, config['links'])

    net, stub_map = remove_stubs(net)

    return net


def cluster_network(net: Network):

    # TODO:
    #  - see cluster_network
    return


if __name__ == '__main__':
    config_ = {"lines": {'s_max_pu': 0.7,
                         'types': {220.: "Al/St 240/40 2-bundle 220.0",
                                   300.: "Al/St 240/40 3-bundle 300.0",
                                   380.: "Al/St 240/40 4-bundle 380.0"}},
               "links": {'p_max_pu': 1.0,
                         'include_tyndp': True},
               "transformers": {}}

    net_ = pypsa.Network()
    voltages_ = [220., 300., 380.]
    countries_ = ["BE", "NL", "LU", "GB"]

    net_ = load_topology(net_, config_, countries_, voltages_)

    net_ = simplify_network(net_)

    print(net_.transformers)

    net_.plot(bus_sizes=0.001)
    import matplotlib.pyplot as plt
    plt.show()

from typing import Tuple

import pandas as pd

from iepy.technologies.manager import get_config_values, get_tech_info, get_fuel_info

from iepy import data_path

NHoursPerYear = 8760.0


def compute_capital_cost(fom: float, capex: float, lifetime: float, nb_hours: float = NHoursPerYear) -> float:
    """
    Compute the 'periodized' cost of building one unit of power of a certain technology.

    Parameters
    ----------
    fom: float
        Fixed operating costs (in currency/power unit*year or currency/power unit*year*length unit)
    capex: float
        Capital expenditure (in currency/power unit or currency/power unit*length unit)
    lifetime: float
        Lifetime (in years)
    nb_hours: float
        Number of hours over which the cost must be periodized

    Returns
    -------
    float
        Periodized cost (in currency/power unit or currency/power unit*length unit)
    """
    return (fom + capex/lifetime) * nb_hours/NHoursPerYear


def compute_marginal_cost(vom: float, fuel_cost: float = 0., efficiency: float = 1.,
                          co2_content: float = 0., co2_cost: float = 0.) -> float:
    """
    Compute the cost of producing/transporting one unit of energy.

    Parameters
    ----------
    vom: float
        Variable operating costs (in currency/unit of electrical energy)
    fuel_cost: float (default: 0.)
        Cost of fuel (in currency/unit of thermal energy)
    efficiency: float (default: 1.)
        Percentage of electrical energy produced using one unit of thermal energy
    co2_content: float (default: 0.)
        Quantity of CO2 (in unit of mass) contained in one unit of thermal energy of fuel
    co2_cost: float (default: 0.)
        Cost of CO2 (in currency/unit of mass)

    Returns
    -------
    float
        Cost of one unit of electrical energy (in currency/unit of electrical energy)
    """
    return vom + (fuel_cost + co2_cost * co2_content) / efficiency


def get_costs(tech: str, nb_hours: float, precision: int = 3) -> Tuple[float, float]:
    """
    Return capital and marginal cost for a given generation technology.

    Parameters
    ----------
    tech: str
        Name of a technology
    nb_hours: float
        Number of hours over which the investment costs will be used
    precision: int (default: 3)
        Indicates at which decimal costs should be rounded

    Returns
    -------
    Capital cost (M€/GWel or M€/GWel/km) and marginal cost (M€/GWhel)

    """

    tech_info = get_tech_info(tech, ["FOM", "CAPEX", "lifetime", "VOM", "efficiency_ds", "fuel"])

    # Capital cost
    capital_cost = compute_capital_cost(tech_info["FOM"], tech_info["CAPEX"], tech_info["lifetime"], nb_hours)

    # Marginal cost
    vom = tech_info['VOM']
    fuel = tech_info['fuel']
    if pd.isna(fuel):
        marginal_cost = compute_marginal_cost(vom)
    else:
        fuel_info = get_fuel_info(fuel, ['cost', 'CO2'])
        co2_cost = get_fuel_info('CO2', ['cost']).values[0]
        marginal_cost = compute_marginal_cost(vom, fuel_info['cost'], tech_info['efficiency_ds'],
                                              fuel_info['CO2'], co2_cost)

    return round(capital_cost, precision), round(marginal_cost, precision)

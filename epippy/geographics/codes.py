from typing import List

import numpy as np
import pandas as pd

import pycountry as pyc

from epippy import data_path


def convert_country_codes(source_codes: List[str], source_format: str, target_format: str,
                          throw_error: bool = False) -> List[str]:
    """
    Convert country codes, e.g., from ISO_2 to full name.

    Parameters
    ----------
    source_codes: List[str]
        List of codes to convert.
    source_format: str
        Format of the source codes (alpha_2, alpha_3, name, ...)
    target_format: str
        Format to which code must be converted (alpha_2, alpha_3, name, ...)
    throw_error: bool (default: False)
        Whether to throw an error if an attribute does not exist.

    Returns
    -------
    target_codes: List[str]
        List of converted codes.
    """
    target_codes = []
    for code in source_codes:
        try:
            country_codes = pyc.countries.get(**{source_format: code})
            if country_codes is None:
                raise KeyError(f"Data is not available for code {code} of type {source_format}.")
            target_code = getattr(country_codes, target_format)
        except (KeyError, AttributeError) as e:
            if throw_error:
                raise e
            target_code = np.nan
        target_codes += [target_code]
    return target_codes


def remove_landlocked_countries(country_list: List[str]) -> List[str]:
    """Filtering out landlocked countries."""
    # landlocked_countries = {'AT', 'BA', 'BY', 'CH', 'CZ', 'HU', 'LI', 'LU', 'MD', 'MK', 'RS', 'SK', 'SI'}
    landlocked_countries = {'AT', 'BY', 'CH', 'CZ', 'HU', 'LI', 'LU', 'MD', 'MK', 'RS', 'SK'}
    return sorted(list(set(country_list) - landlocked_countries))


def get_subregions(region: str) -> List[str]:
    """
    Return the list of the subregions composing one of the region defined in 'data_path'/geographics/region_definition.csv.

    Parameters
    ----------
    region: str
        Code of a geographical region defined in 'data_path'/geographics/region_definition.csv.

    Returns
    -------
    subregions: List[str]
        List of subregion codes, if no subregions, returns [region]
    """

    region_definition_fn = f"{data_path}geographics/region_definition.csv"
    region_definition = pd.read_csv(region_definition_fn, index_col=0, keep_default_na=False)

    if region in region_definition.index:
        subregions = region_definition.loc[region].subregions.split(";")
    else:
        subregions = [region]

    return subregions


def replace_iso2_codes(countries_list: List[str]) -> List[str]:
    """
    Updating ISO_2 code for UK and EL (not uniform across datasets).

    Parameters
    ----------
    countries_list: List[str]
        Initial list of ISO_2 codes.

    Returns
    -------
    updated_codes: List[str]
        Updated ISO_2 codes.
    """

    country_names_issues = {'UK': 'GB', 'EL': 'GR', 'KV': 'XK'}
    updated_codes = [country_names_issues[c] if c in country_names_issues else c for c in countries_list]

    return updated_codes


def revert_iso2_codes(countries_list: List[str]) -> List[str]:
    """
    Reverting ISO_2 code for UK and EL (not uniform across datasets).

    Parameters
    ----------
    countries_list: List[str]
        Initial list of ISO_2 codes.

    Returns
    -------
    updated_codes: List[str]
        Updated ISO_2 codes.
    """

    country_names_issues = {'GB': 'UK', 'GR': 'EL', 'XK': 'KV'}
    updated_codes = [country_names_issues[c] if c in country_names_issues else c for c in countries_list]

    return updated_codes


def convert_old_country_names(c: str) -> str:
    """Converting country old full names to new ones, as some datasets are not updated on the issue."""

    if c == "Macedonia":
        return "North Macedonia"

    if c == "Czech Republic":
        return "Czechia"

    if c == 'Syria':
        return 'Syrian Arab Republic'

    if c == 'Iran':
        return 'Iran, Islamic Republic of'

    if c == "Byelarus":
        return "Belarus"

    return c


def revert_old_country_names(c: str) -> str:
    """Reverting country full names to old ones, as some datasets are not updated on the issue."""

    if c == "North Macedonia":
        return "Macedonia"

    if c == "Czechia":
        return "Czech Republic"

    return c


def get_nuts_codes(nuts_level: int, year: int, countries: List[str] = None):
    available_years = [2013, 2016]
    assert year in available_years, f"Error: Year must be one of {available_years}, received {year}"
    available_nuts_levels = [0, 1, 2, 3]
    assert nuts_level in available_nuts_levels, \
        f"Error: NUTS level must be one of {available_nuts_levels}, received {nuts_level}"

    nuts_fn = f"{data_path}geographics/source/eurostat/NUTS2013-NUTS2016.xlsx"
    nuts_codes = pd.read_excel(nuts_fn, sheet_name="NUTS2013-NUTS2016", usecols=[1, 2], header=1)
    nuts_codes = nuts_codes[f"Code {year}"].dropna().tolist()

    # Norway
    nuts_codes += ['NO', 'NO0',
                   'NO01', 'NO011', 'NO012',
                   'NO02', 'NO021', 'NO022',
                   'NO03', 'NO031', 'NO032', 'NO033', 'NO034',
                   'NO04', 'NO041', 'NO042', 'NO043',
                   'NO05', 'NO051', 'NO052', 'NO053',
                   'NO06', 'NO061', 'NO062',
                   'NO07', 'NO071', 'NO072', 'NO073']
    # Switzerland
    nuts_codes += ['CH', 'CH0',
                   'CH01', 'CH011', 'CH012', 'CH013',
                   'CH02', 'CH021', 'CH022', 'CH023', 'CH024', 'CH025',
                   'CH03', 'CH031', 'CH032', 'CH033',
                   'CH04', 'CH040',
                   'CH05', 'CH051', 'CH052', 'CH053', 'CH054', 'CH055', 'CH056', 'CH057',
                   'CH06', 'CH061', 'CH062', 'CH063', 'CH064', 'CH065', 'CH066',
                   'CH07', 'CH070']

    # Montenegro
    nuts_codes += ['ME0', 'ME00', 'ME000']

    # Serbia
    nuts_codes += ['RS', 'RS1', 'RS2',
                   'RS11', 'RS110',
                   'RS12', 'RS121', 'RS122', 'RS123', 'RS124', 'RS125', 'RS126', 'RS127',
                   'RS21', 'RS211', 'RS212', 'RS213', 'RS214', 'RS215', 'RS216', 'RS217', 'RS218',
                   'RS22', 'RS221', 'RS222', 'RS223', 'RS224', 'RS225', 'RS226', 'RS227', 'RS228', 'RS229']

    # North Macedonia
    nuts_codes += ['MK', 'MK0', 'MK00',
                   'MK001', 'MK002', 'MK003', 'MK004', 'MK005', 'MK006', 'MK007', 'MK008']

    # Albania
    nuts_codes += ['AL', 'AL0',
                   'AL01', 'AL011', 'AL012', 'AL013', 'AL014', 'AL015',
                   'AL02', 'AL021', 'AL022',
                   'AL03', 'AL031', 'AL032', 'AL033', 'AL034', 'AL035']

    # Bosnia
    # nuts_codes += ['BA', 'BA0', 'BA00', 'BA000']

    # Get NUTS at the right level
    nuts_codes = [code for code in nuts_codes if len(code) == nuts_level + 2]

    # Filter on countries
    if countries is not None:
        nuts_codes = [code for code in nuts_codes if code[:2] in countries]

    return nuts_codes

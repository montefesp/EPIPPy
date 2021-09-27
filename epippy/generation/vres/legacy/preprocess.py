from os.path import join
from typing import List
import warnings

import pandas as pd
import xarray as xr
import numpy as np

from epippy.geographics.grid_cells import get_grid_cells
from epippy.geographics import get_shapes, match_points_to_regions, convert_country_codes
from epippy.technologies import get_config_values

from epippy import data_path


def get_legacy_capacity_in_regions_from_non_open(tech: str, regions_shapes: pd.Series, countries: List[str],
                                                 spatial_res: float, include_operating: bool,
                                                 match_distance: float = 50., raise_error: bool = True) -> pd.Series:
    """
    Return the total existing capacity (in GW) for the given tech for a set of geographical regions.

    This function is using proprietary data.

    Parameters
    ----------
    tech: str
        Technology name.
    regions_shapes: pd.Series [Union[Polygon, MultiPolygon]]
        Geographical regions
    countries: List[str]
        List of ISO codes of countries in which the regions are situated
    spatial_res: float
        Spatial resolution of data
    include_operating: bool
        Include or not the legacy capacity of already operating units.
    match_distance: float (default: 50)
        Distance threshold (in km) used when associating points to shape.
    raise_error: bool (default: True)
        Whether to raise an error if no legacy data is available for this technology.

    Returns
    -------
    capacities: pd.Series
        Legacy capacities (in GW) of technology 'tech' for each region

    """

    path_legacy_data = f"{data_path}generation/vres/legacy/source/"
    path_gdp_data = f"{data_path}indicators/gdp/source"
    path_pop_data = f"{data_path}indicators/population/source"

    capacities = pd.Series(0., index=regions_shapes.index)
    plant, plant_type = get_config_values(tech, ["plant", "type"])
    if (plant, plant_type) in [("Wind", "Onshore"), ("Wind", "Offshore"), ("PV", "Utility")]:

        if plant == "Wind":

            data = pd.read_excel(f"{path_legacy_data}Windfarms_Europe_20200127.xls", sheet_name='Windfarms',
                                 header=0, usecols=[2, 5, 9, 10, 18, 22, 23], skiprows=[1], na_values='#ND')
            data = data.dropna(subset=['Latitude', 'Longitude', 'Total power'])

            if include_operating:
                plant_status = ['Planned', 'Approved', 'Construction', 'Production']
            else:
                plant_status = ['Planned', 'Approved', 'Construction']
            data = data.loc[data['Status'].isin(plant_status)]

            if countries is not None:
                data = data[data['ISO code'].isin(countries)]

            if len(data) == 0:
                return capacities

            # Converting from kW to GW
            data['Total power'] *= 1e-6
            data["Location"] = data[["Longitude", "Latitude"]].apply(lambda x: (x.Longitude, x.Latitude), axis=1)

            # Keep only onshore or offshore point depending on technology
            if plant_type == 'Onshore':
                data = data[data['Area'] != 'Offshore']
            else:  # Offshore
                data = data[data['Area'] == 'Offshore']

            if len(data) == 0:
                return capacities

        else:  # plant == "PV":

            data = pd.read_excel(f"{path_legacy_data}Solarfarms_Europe_20200208.xlsx", sheet_name='ProjReg_rpt',
                                 header=0, usecols=[0, 4, 5, 6, 8])
            data = data[pd.notnull(data['Coords'])]

            if include_operating:
                plant_status = ['Building', 'Planned', 'Active']
            else:
                plant_status = ['Building', 'Planned']
            data = data.loc[data['Status'].isin(plant_status)]

            data["Location"] = data["Coords"].apply(lambda x: (float(x.split(',')[1]), float(x.split(',')[0])))
            if countries is not None:
                data['Country'] = convert_country_codes(data['Country'].values, 'name', 'alpha_2')
                data = data[data['Country'].isin(countries)]

            if len(data) == 0:
                return capacities

            # Converting from MW to GW
            data['Total power'] = data['MWac']*1e-3

        data = data[["Location", "Total power"]]

        points_region = match_points_to_regions(data["Location"].values, regions_shapes,
                                                distance_threshold=match_distance).dropna()

        for region in regions_shapes.index:
            points_in_region = points_region[points_region == region].index.values
            capacities[region] = data[data["Location"].isin(points_in_region)]["Total power"].sum()

    elif (plant, plant_type) == ("PV", "Residential"):

        legacy_capacity_fn = join(path_legacy_data, 'SolarEurope_Residential_deployment.xlsx')
        data = pd.read_excel(legacy_capacity_fn, header=0, index_col=0, usecols=[0, 4], squeeze=True).sort_index()
        data = data[data.index.isin(countries)]

        if len(data) == 0:
            return capacities

        # TODO: where is this file ?
        gdp_data_fn = join(path_gdp_data, "GDP_per_capita_PPP_1990_2015_v2.nc")
        gdp_data = xr.open_dataset(gdp_data_fn)
        gdp_2015 = gdp_data.sel(time='2015.0')

        pop_data_fn = join(path_pop_data, "gpw_v4_population_count_adjusted_rev11_15_min.nc")
        pop_data = xr.open_dataset(pop_data_fn)
        pop_2020 = pop_data.sel(raster=5)

        # Temporary, to reduce the size of this ds, which is anyway read in each iteration.
        min_lon, max_lon, min_lat, max_lat = -11., 32., 35., 80.
        mask_lon = (gdp_2015.longitude >= min_lon) & (gdp_2015.longitude <= max_lon)
        mask_lat = (gdp_2015.latitude >= min_lat) & (gdp_2015.latitude <= max_lat)

        new_lon = np.arange(min_lon, max_lon+spatial_res, spatial_res)
        new_lat = np.arange(min_lat, max_lat+spatial_res, spatial_res)

        gdp_ds = gdp_2015.where(mask_lon & mask_lat, drop=True)['GDP_per_capita_PPP']
        pop_ds = pop_2020.where(mask_lon & mask_lat, drop=True)['UN WPP-Adjusted Population Count, v4.11 (2000,'
                                                                ' 2005, 2010, 2015, 2020): 15 arc-minutes']

        gdp_ds = gdp_ds.reindex(longitude=new_lon, latitude=new_lat, method='nearest')\
            .stack(locations=('longitude', 'latitude'))
        pop_ds = pop_ds.reindex(longitude=new_lon, latitude=new_lat, method='nearest')\
            .stack(locations=('longitude', 'latitude'))

        all_sites = [(idx[0], idx[1]) for idx in regions_shapes.index]
        total_gdp_per_capita = gdp_ds.sel(locations=all_sites).sum().values
        total_population = pop_ds.sel(locations=all_sites).sum().values

        df_metrics = pd.DataFrame(index=regions_shapes.index, columns=['gdp', 'pop'])
        for region_id, region_shape in regions_shapes.items():
            lon, lat = region_id[0], region_id[1]
            df_metrics.loc[region_id, 'gdp'] = gdp_ds.sel(longitude=lon, latitude=lat).values/total_gdp_per_capita
            df_metrics.loc[region_id, 'pop'] = pop_ds.sel(longitude=lon, latitude=lat).values/total_population

        df_metrics['gdppop'] = df_metrics['gdp'] * df_metrics['pop']
        df_metrics['gdppop_norm'] = df_metrics['gdppop']/df_metrics['gdppop'].sum()

        capacities = df_metrics['gdppop_norm'] * data[countries[0]]
        capacities = capacities.reset_index()['gdppop_norm']

    else:
        if raise_error:
            raise ValueError(f"Error: No legacy data exists for tech {tech} with plant {plant} and type {plant_type}.")
        else:
            warnings.warn(f"Warning: No legacy data exists for tech {tech}.")

    return capacities.astype(float)


def aggregate_legacy_capacity(spatial_resolution: float, include_operating: bool):
    """
    Aggregate legacy data at a given spatial resolution.

    Parameters
    ----------
    spatial_resolution: float
        Spatial resolution at which we want to aggregate.
    include_operating: bool
        Whether to include already operating plants or not.

    """

    countries = ["AL", "AT", "BA", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES",
                 "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LT", "LU", "LV",
                 "ME", "MK", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI", "SK"] # removed ["BY", "UA", "FO"]

    technologies = ["wind_onshore", "wind_offshore", "pv_utility", "pv_residential"]

    capacities_df_ls = []
    for country in countries:
        print(f"Country: {country}")
        shapes = get_shapes([country])
        onshore_shape = shapes[~shapes["offshore"]]["geometry"]
        offshore_shape = shapes[shapes["offshore"]]["geometry"]
        # If not offshore shape for country, remove offshore technologies from set
        offshore_shape = None if len(offshore_shape) == 0 else offshore_shape
        technologies_in_country = technologies
        if offshore_shape is None:
            technologies_in_country = [tech for tech in technologies if get_config_values(tech, ['onshore'])]

        # Divide shapes into grid cells
        grid_cells_ds = get_grid_cells(technologies_in_country, spatial_resolution, onshore_shape, offshore_shape)
        technologies_in_country = set(grid_cells_ds.index.get_level_values(0))

        # Get capacity in each grid cell
        capacities_per_country_ds = pd.Series(index=grid_cells_ds.index, name="Capacity (GW)", dtype=float)
        for tech in technologies_in_country:

            idx = capacities_per_country_ds[tech].index
            if tech == 'pv_residential':
                capacities_per_country_and_tech = \
                    get_legacy_capacity_in_regions_from_non_open(tech, grid_cells_ds.loc[tech], [country],
                                                                 spatial_resolution, include_operating,
                                                                 match_distance=100)
            else:
                capacities_per_country_and_tech = \
                    get_legacy_capacity_in_regions_from_non_open(tech, grid_cells_ds.loc[tech].reset_index()[0],
                                                                 [country], spatial_resolution, include_operating,
                                                                 match_distance=100)
            capacities_per_country_and_tech.index = idx
            capacities_per_country_ds[tech].update(capacities_per_country_and_tech)

        capacities_per_country_df = capacities_per_country_ds.to_frame()
        capacities_per_country_df.loc[:, "ISO2"] = country
        capacities_df_ls += [capacities_per_country_df]

    # Aggregate dataframe from each country
    capacities_df = pd.concat(capacities_df_ls).sort_index()

    # Replace technology name by plant and type
    tech_to_plant_type = {tech: get_config_values(tech, ["plant", "type"]) for tech in technologies}
    capacities_df = capacities_df.reset_index()
    capacities_df["Plant"] = capacities_df["Technology Name"].apply(lambda x: tech_to_plant_type[x][0])
    capacities_df["Type"] = capacities_df["Technology Name"].apply(lambda x: tech_to_plant_type[x][1])
    capacities_df = capacities_df.drop("Technology Name", axis=1)
    capacities_df = capacities_df.set_index(["Plant", "Type", "Longitude", "Latitude"])

    legacy_dir = f"{data_path}generation/vres/legacy/generated/"
    capacities_df.round(4).to_csv(f"{legacy_dir}aggregated_capacity.csv",
                                  header=True, columns=["ISO2", "Capacity (GW)"])


if __name__ == '__main__':

    aggregate_legacy_capacity(spatial_resolution=0.25, include_operating=True)

    # technologies = ["pv_residential"]
    # spatial_res = 1.0
    # countries = ["DE", "FR", "BE"]
    # cap_dict = dict.fromkeys(countries)
    #
    # for country in countries:
    #
    #     shapes = get_shapes([country], which='onshore')
    #     onshore_shape = shapes["geometry"].values[0]
    #     region_shapes = get_grid_cells(technologies, spatial_res, onshore_shape)
    #
    #     for tech in technologies:
    #
    #         cap_dict[country] = get_legacy_capacity_in_regions_from_non_open(tech, region_shapes, [country], spatial_res)
    #         print(cap_dict[country])


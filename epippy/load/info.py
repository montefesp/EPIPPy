import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from epippy import data_path


def available_load_profiles():


    # 2019 data

    source_data_fn = f"{data_path}load/source/time_series_60min_singleindex_filtered_2019.csv"
    source_data = pd.read_csv(source_data_fn, index_col='utc_timestamp')
    source_data = source_data.drop(["cet_cest_timestamp"], axis=1)
    source_data = source_data[1:-23]
    print(source_data.index)

    # Selecting first the keys from entsoe transparency platform
    all_keys = [key for key in source_data.keys() if 'load_actual_entsoe_transparency' in key]

    # Adding then the keys from power statistics corresponding to missing countries
    all_keys_short = [key.split("_load")[0] for key in all_keys]
    all_keys += [key for key in source_data.keys()
                 if 'load_actual_entsoe_power' in key and key.split("_load")[0] not in all_keys_short]

    # Finally add the keys from tso corresponding to other missing countries
    all_keys_short = [key.split("_load")[0] for key in all_keys]
    all_keys += [key for key in source_data.keys()
                 if 'load_actual_tso' in key and key.split("_load")[0] not in all_keys_short]

    final_data = source_data[all_keys]
    final_data.columns = [key.split("_load")[0] for key in final_data.keys()]

    # Remove some shitty data by inspection
    final_data = final_data.drop(["CS", "IE_sem", "DE_LU", "GB_NIR", "GB_UKM"], axis=1)

    # Change GB_GBN to GB
    final_data = final_data.rename(columns={"GB_GBN": "GB"})

    # Change index to pandas.DatetimeIndex
    final_data.index = pd.DatetimeIndex(final_data.index)

    final_data = final_data.reindex(sorted(final_data.columns), axis=1)

    print((final_data.isna().sum()/len(final_data)).round(2))
    first_last_valid_index = pd.concat((final_data.notna().idxmax(), final_data.notna()[::-1].idxmax()), axis=1)
    print(first_last_valid_index)
    # first_last_valid_index.plot(linestyle='-')
    # plt.show()

    zones = final_data.columns
    availability_matrix = pd.DataFrame(0, columns=range(2005, 2020), index=zones, dtype=int)
    validity_ds = pd.Series(0., index=zones, dtype=float)
    for c in zones:
        # Compute full years of data between first and last valid indexes
        start = first_last_valid_index.loc[c, 0]
        year_start = start.year
        if not (start.month == 1 and start.day == 1 and start.hour == 0):
            year_start += 1
        end = first_last_valid_index.loc[c, 1]
        year_end = end.year
        if not (end.month == 12 and end.day == 31 and end.hour == 23):
            year_end -= 1
        availability_matrix.loc[c, range(year_start, year_end+1)] = 1

        # Compute what percentage of data is valid between those two index
        valid_timespan = final_data.loc[first_last_valid_index.loc[c, 0]:first_last_valid_index.loc[c, 1], c]
        validity_ds[c] = (valid_timespan.isna().sum()/len(valid_timespan)).round(2)

    plt.figure()
    validity_ds[validity_ds != 0].sort_index(ascending=False).plot(kind='barh')

    fig, ax = plt.subplots(1, 1)
    ax.imshow(availability_matrix.values, cmap='Blues')

    # Major ticks
    ax.set_xticks(range(0, 15))
    ax.set_yticks(range(0, len(zones)))

    # Labels for major ticks
    ax.set_xticklabels(range(2005, 2020))
    ax.set_yticklabels(zones)

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, 15, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(zones), 1), minor=True)

    plt.xticks(rotation=90)
    ax.grid(which='minor', color='w', ls='-', lw=2)
    plt.title("Full OPSD hourly load data years")


    # 2020 data
    source_data_fn = f"{data_path}load/source/time_series_60min_singleindex_filtered_2019.csv"
    source_data = pd.read_csv(source_data_fn, index_col='utc_timestamp')
    source_data = source_data.drop(["cet_cest_timestamp"], axis=1)
    source_data = source_data[1:-23]
    print(source_data.index)

    plt.show()


if __name__ == '__main__':
    available_load_profiles()

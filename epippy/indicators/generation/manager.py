import pandas as pd


def compute_capacity_credit_from_potential(load_df: pd.DataFrame, cf_df: pd.DataFrame,
                                           tpr_ds: pd.Series, peak_sample: float = 0.01):
    """
    Compute capacity credit based on Milligan eq. from the CF dataframe of all candidate sites.

    Parameters:
    ------------
    load_df: pd.DataFrame
        Load time series frame.
    cf_df: pd.DataFrame
        Capacity factors frame.
    tpr_ds: pd.Series
        Series containing the (technology, lon, lat, country) tuple.
    peak_sample : float (default: 0.01, 1%)
        The top % wrt which the capacity credit is computed.
    """
    cc_df = pd.Series(index=tpr_ds.index, dtype=float)
    nvals = int(peak_sample*len(load_df.index))

    for c in load_df.columns:

        load_data = load_df.loc[:, c].squeeze()
        load_data_peak_index = load_data.nlargest(nvals).index

        gens = tpr_ds.index[(tpr_ds == c)]
        cc_df.loc[gens] = cf_df.loc[load_data_peak_index, gens].mean()

    cc_df = cc_df.reset_index()
    cc_df.index = cc_df['Technology Name'] + ' ' + cc_df['Longitude'].astype(str) + '-' + cc_df['Latitude'].astype(str)
    cc_df.columns = ['Technology Name', 'Longitude', 'Latitude', 'CC']

    return cc_df['CC'].dropna()



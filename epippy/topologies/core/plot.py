import pandas as pd

import matplotlib.pyplot as plt


def plot_topology(buses: pd.DataFrame, lines: pd.DataFrame = None) -> None:
    """
    Plot a map with buses and lines.

    Parameters
    ----------
    buses: pd.DataFrame
        DataFrame with columns 'x', 'y' and 'region'
    lines: pd.DataFrame (default: None)
        DataFrame with columns 'bus0', 'bus1' whose values must be index of 'buses'.
        If None, do not display the lines.
    """

    shapes = buses.onshore_region.dropna()
    if hasattr(buses, 'offshore_region'):
        shapes = pd.concat([shapes, buses.offshore_region.dropna()])
    from epippy.geographics.plot import display_polygons
    ax = display_polygons(shapes.values, show=False)

    # Plotting the buses
    for idx in buses.index:
        # Plot the bus position
        ax.scatter(buses.loc[idx].x, buses.loc[idx].y, c='grey', marker="o", s=10)

    # Plotting the lines
    if lines is not None:
        for idx in lines.index:

            bus0 = lines.loc[idx].bus0
            bus1 = lines.loc[idx].bus1
            if bus0 not in buses.index or bus1 not in buses.index:
                print(f"Warning: not showing line {idx} because missing bus {bus0} or {bus1}")
                continue

            color = 'darkred' if 'carrier' in lines.columns and lines.loc[idx].carrier == "DC" else 'navy'
            plt.plot([buses.loc[bus0].x, buses.loc[bus1].x], [buses.loc[bus0].y, buses.loc[bus1].y], c=color, alpha=0.5)

    return ax

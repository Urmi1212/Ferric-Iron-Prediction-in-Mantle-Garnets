import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

from matplotlib.colors import Normalize


def load_sample_data():
    """Loads sample garnet data with lat/lon/depth/ΔFMQ."""
    return pd.DataFrame({
        'latitude': [15.0, -25.4, 40.2, 60.5, -10.1, 35.2],
        'longitude': [-120.0, 140.2, -75.6, 10.0, 30.0, -60.4],
        'depth_km': [150, 100, 120, 200, 80, 140],
        'delta_FMQ': [-2.0, -1.0, 0.5, 1.5, -0.5, 2.0]
    })


def group_data(df, precision=1):
    """Rounds coordinates and aggregates data by location."""
    df['lat_rounded'] = df['latitude'].round(precision)
    df['lon_rounded'] = df['longitude'].round(precision)
    grouped = df.groupby(['lat_rounded', 'lon_rounded']).agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'depth_km': 'mean',
        'delta_FMQ': 'mean'
    }).reset_index(drop=True)
    return grouped


def plot_map(df, title="Global Mantle Redox Conditions (ΔFMQ)", save_path=None):
    """Plots a map of ΔFMQ data."""
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True)

    norm = Normalize(vmin=-2.5, vmax=2.5)
    cmap = plt.get_cmap('coolwarm')

    scatter = ax.scatter(
        df['longitude'], df['latitude'],
        c=df['delta_FMQ'], cmap=cmap, norm=norm,
        s=100, edgecolors='k', transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('ΔFMQ (Relative Oxygen Fugacity)', fontsize=12)
    plt.title(title, fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def get_group_counts(df, precision=1):
    """Returns count of data points in each rounded location group."""
    df['group_id'] = list(zip(df['latitude'].round(precision), df['longitude'].round(precision)))
    return df['group_id'].value_counts()


# Optional test run
if __name__ == "__main__":
    df = load_sample_data()
    df_grouped = group_data(df)
    plot_map(df_grouped)
    counts = get_group_counts(df)
    print(counts)

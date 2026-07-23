from pathlib import Path
from glob import glob

import cartopy.crs as ccrs
import cmocean as cmo

from phd_ocean.eda import (
    load_many_datasets,
    run_eda_1d,
    run_eda_2d,
    run_eda_2d_anomalies,
    select_region,
    transform_longitude,
)


# ---------------------------------------------------------------------
# Paths and variable settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = Path("/home/paulo/Documents/data/atmospheric/JRA3Q")
DATA_FILES = sorted(glob(str(DATA_PATH / "t2*")))

TEMP_VAR = "t2m"
TIME_DIM = "time"
DEPTH_INDEX = None

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "jra3q_t2m"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
NETCDF_DIR = OUTPUT_ROOT / "netcdf"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
NETCDF_DIR.mkdir(parents=True, exist_ok=True)

CLIMATOLOGY_YEARS = (1991, 2020)


# ---------------------------------------------------------------------
# Region settings
# ---------------------------------------------------------------------

EXTENT = [-100, -50, 50, 85]


def cut_region(ds):
    return select_region(
        ds,
        lat_min=50,
        lat_max=85,
        lon_min=-100,
        lon_max=-50,
        lat_name="lat",
        lon_name="lon",
    )


# ---------------------------------------------------------------------
# Projection and plotting settings
# ---------------------------------------------------------------------

projection = ccrs.AlbersEqualArea(
    central_longitude=-80,
    central_latitude=60,
    standard_parallels=(50, 70),
)

base_plot_kwargs = {
    "projection": projection,
    "data_crs": ccrs.PlateCarree(),
    "extent": EXTENT,
    "levels": 30,
    "title": None,
    "colorbar_tick_rotation": 20,
    "colorbar_scientific": True,
    "colorbar_tick_size": 9,
    "colorbar_label_size": 12,
    "colorbar_label_pad": 5,
    "map_features": {
        "land": False,
        "coastline": True,
        "borders": False,
        "states": False,
        "draw_labels": True,
        "gridline_step": 5,
        "gridline_crs": ccrs.PlateCarree(),
        "gridline_kwargs": {
            "linewidth": 0.7,
            "color": "black",
            "alpha": 0.15,
            "linestyle": "--",
            "zorder": 5,
        },
    },
}

stat_plot_kwargs = {
    **base_plot_kwargs,
    "colorbar_label": "2m temperature (K)",
}

period_plot_kwargs = {
    **base_plot_kwargs,
    "cmap": cmo.cm.thermal,
    "centered": False,
    "colorbar_label": "2m temperature (K)",
}

anomaly_plot_kwargs = {
    "cmap": cmo.cm.balance,
    "centered": True,
    "colorbar_label": "2m temperature anomaly (K)",
    "colorbar_scientific": True,
}


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main():
    if not DATA_FILES:
        raise FileNotFoundError(f"No JRA3Q files found in {DATA_PATH}")

    print(f"Loading {len(DATA_FILES)} files for {TEMP_VAR}...")

    ds = load_many_datasets(
        DATA_FILES,
        concat_dim=TIME_DIM,
        transform=transform_longitude,
        transform_second=cut_region,
    )

    print(f"Running 1D EDA for {TEMP_VAR}...")

    temp_summary = run_eda_1d(
        ds,
        variable=TEMP_VAR,
        time_dim=TIME_DIM,
        depth_index=DEPTH_INDEX,
        output_table=TABLE_DIR / "jra3q_t2m_1d_summary.csv",
        output_violin=FIGURE_DIR / "jra3q_t2m_violin.png",
        ylabel="2m temperature (K)",
        title="JRA3Q 2m temperature yearly distribution",
        show=False,
    )

    print(temp_summary)

    print("Running 2D full-period statistics for temperature...")

    run_eda_2d(
        ds,
        variable=TEMP_VAR,
        time_dim=TIME_DIM,
        depth_index=DEPTH_INDEX,
        mode="statistics",
        statistics=["mean", "std", "q50", "iqr", "yki", "kurtosis"],
        output_dir=FIGURE_DIR / "statistics",
        output_statistics_nc=NETCDF_DIR / "jra3q_t2m_statistics.nc",
        lon_name="lon",
        lat_name="lat",
        plot_kwargs=stat_plot_kwargs,
        show=False,
    )

    print("Running 2D monthly/seasonal/annual mean maps for temperature...")

    run_eda_2d(
        ds,
        variable=TEMP_VAR,
        time_dim=TIME_DIM,
        depth_index=DEPTH_INDEX,
        mode="periods",
        periods=["monthly", "seasonal", "annual"],
        reducer="mean",
        output_dir=FIGURE_DIR / "periods",
        save_periods_netcdf=True,
        lon_name="lon",
        lat_name="lat",
        plot_kwargs=period_plot_kwargs,
        show=False,
    )

    print(
        "Running monthly, seasonal, and yearly climatological anomalies "
        f"relative to {CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[1]}..."
    )

    run_eda_2d_anomalies(
        ds,
        variable=TEMP_VAR,
        output_dir=OUTPUT_ROOT,
        anomaly_periods=["monthly", "seasonal", "yearly"],
        climatology_years=CLIMATOLOGY_YEARS,
        time_dim=TIME_DIM,
        depth_index=DEPTH_INDEX,
        plot_kwargs=base_plot_kwargs,
        anomaly_plot_kwargs=anomaly_plot_kwargs,
        write_netcdf=True,
        show=False,
    )

    print("Done.")


if __name__ == "__main__":
    main()
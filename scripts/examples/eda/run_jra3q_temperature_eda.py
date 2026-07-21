from pathlib import Path

from phd_ocean.eda import run_eda_1d, run_eda_2d
from phd_ocean.eda import select_region, transform_longitude
from phd_ocean.eda import load_many_datasets
import xarray as xr
from glob import glob
import cartopy.crs as ccrs

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


#DATA_FILE = Path("/home/paulo/Documents/data/atmospheric/JRA3Q/q2_JRA3Q_y1958.nc")

DATA_PATHS = "/home/paulo/Documents/data/atmospheric/JRA3Q/"
DATA_FILES = sorted(glob(DATA_PATHS+"q2*"))

# Change these after running inspect_jra3q_dataset.py
TEMP_VAR = "q2m"
TIME_DIM = "time"

# Use None if your file has no depth dimension.
DEPTH_INDEX = None

# Change for your region if needed.
EXTENT = [-100, -50, 45, 85]

ds = load_many_datasets(DATA_FILES,
                        concat_dim=TIME_DIM,
                        transform=transform_longitude,
                        transform_second=cut_region)

da = ds[TEMP_VAR]
# da = transform_longitude(da)
# da = select_region(da,
#                    lat_min = 45,
#                    lat_max = 72,
#                    lon_min = -105,
#                    lon_max = -60)



projection = ccrs.AlbersEqualArea(
    central_longitude=-80,
    central_latitude=60,
    standard_parallels=(50, 70),
)

plot_kwargs = {
    "projection": projection,
    "data_crs": ccrs.PlateCarree(),
    "extent": EXTENT,
    "levels": 30,
    "cmap": "viridis",
    "title": None,
    "colorbar_label": "Specific humidity",
    "gridline_step": 5,
    "map_features": {
        "land": False,
        "coastline": True,
        "borders": False,
        "states": False,
        "draw_labels": True,
        "gridline_crs": ccrs.PlateCarree(),
    },
}

def main():
    print("Running 1D EDA for "+TEMP_VAR+"...")

    temp_summary = run_eda_1d(
        da,
        variable=TEMP_VAR,
        time_dim=TIME_DIM,
        depth_index=DEPTH_INDEX,
        output_table="/home/paulo/Documents/codes/mine/PhD_codes/python/phd-ocean-analysis/outputs/tables/jra3q_SH_1d_summary.csv",
        output_violin="/home/paulo/Documents/codes/mine/PhD_codes/python/phd-ocean-analysis/outputs/figures/jra3q_SH_violin.png",
        ylabel="2m specific humidity",
        title="JRA3Q SH yearly distribution",
        show=False,
    )

    print(temp_summary)

    print("Running 2D full-period statistics for temperature...")

    run_eda_2d(
        da,
        variable=TEMP_VAR,
        time_dim=TIME_DIM,
        depth_index=DEPTH_INDEX,
        mode="statistics",
        statistics=["mean", "std", "q50", "iqr", "yki", "kurtosis"],
        output_dir="/home/paulo/Documents/codes/mine/PhD_codes/python/phd-ocean-analysis/outputs/figures/jra3q_specific_humidity/statistics",
        output_statistics_nc="/home/paulo/Documents/codes/mine/PhD_codes/python/phd-ocean-analysis/outputs/netcdf/jra3q_specifichumidity_statistics.nc",
        lon_name="lon",
        lat_name="lat",
        plot_kwargs=plot_kwargs,
        show=False,
    )

    print("Running 2D seasonal/monthly/annual maps for temperature...")

    run_eda_2d(
        da,
        variable=TEMP_VAR,
        time_dim=TIME_DIM,
        depth_index=DEPTH_INDEX,
        mode="periods",
        periods=["seasonal", "monthly", "annual"],
        reducer="mean",
        output_dir="/home/paulo/Documents/codes/mine/PhD_codes/python/phd-ocean-analysis/outputs/figures/jra3q_specific_humidity/periods",
        save_periods_netcdf=True,
        lon_name="lon",
        lat_name="lat",
        plot_kwargs=plot_kwargs,
        show=False,
    )

    print("Done.")


if __name__ == "__main__":
    main()
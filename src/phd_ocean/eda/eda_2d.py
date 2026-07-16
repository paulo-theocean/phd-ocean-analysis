from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import xarray as xr

from .io import (
    expand_paths,
    get_first_data_variable,
    load_dataset,
    load_many_datasets,
)
from .plotting import get_lon_lat, plot_2d_map, plot_2d_subplots
from .stats import DEFAULT_2D_STATS, aggregate_periods, compute_2d_statistics


PathLike = str | Path
XarrayObject = xr.Dataset | xr.DataArray

Mode = Literal["statistics", "periods", "both"]
Period = Literal["full", "seasonal", "monthly", "annual"]


DEPTH_DIM_CANDIDATES = (
    "depth",
    "deptht",
    "depthu",
    "depthv",
    "depthw",
    "lev",
    "z",
)


def _infer_depth_dim(data: xr.DataArray | xr.Dataset) -> str | None:
    """
    Infer a likely vertical/depth dimension name.
    """
    for name in DEPTH_DIM_CANDIDATES:
        if name in data.dims:
            return name

    for dim in data.dims:
        dim_lower = str(dim).lower()
        if "depth" in dim_lower or dim_lower in {"z", "lev"}:
            return str(dim)

    return None


def _open_input_data(
    data: XarrayObject | PathLike | Sequence[PathLike],
    *,
    concat_dim: str | None = None,
    engine: str | None = "netcdf4",
    chunks: dict[str, Any] | None = None,
) -> XarrayObject:
    """
    Open input data when a path, glob pattern, or list of paths is provided.

    If an xarray Dataset/DataArray is provided, it is returned unchanged.
    """
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        return data

    paths = expand_paths(data)

    if len(paths) == 1:
        return load_dataset(
            paths[0],
            engine=engine,
            chunks=chunks,
        )

    if concat_dim is None:
        raise ValueError(
            "Multiple input files were provided, so concat_dim is required. "
            "For example: concat_dim='time' or concat_dim='time_counter'."
        )

    return load_many_datasets(
        paths,
        concat_dim=concat_dim,
        engine=engine,
        chunks=chunks,
    )


def select_variable(
    data: XarrayObject,
    *,
    variable: str | None = None,
) -> xr.DataArray:
    """
    Select one variable from a Dataset or return the DataArray unchanged.
    """
    if isinstance(data, xr.DataArray):
        return data

    if not isinstance(data, xr.Dataset):
        raise TypeError("data must be an xarray Dataset or DataArray.")

    selected_variable = variable or get_first_data_variable(data)

    if selected_variable not in data.data_vars:
        raise KeyError(
            f"Variable {selected_variable!r} was not found. "
            f"Available variables: {list(data.data_vars)}"
        )

    return data[selected_variable]


def prepare_2d_field(
    data: XarrayObject,
    *,
    variable: str | None = None,
    depth_index: int | None = None,
    depth_dim: str | None = None,
    fill_values: Sequence[float] | None = None,
    valid_min: float | None = None,
    valid_max: float | None = None,
) -> xr.DataArray:
    """
    Prepare one variable for EDA 2D analysis.

    Steps
    -----
    1. Select one variable.
    2. Optionally select one vertical level.
    3. Optionally mask fill values and valid ranges.

    Returns
    -------
    xr.DataArray
        Cleaned DataArray ready for 2D statistics or period aggregation.
    """
    da = select_variable(data, variable=variable)

    if depth_index is not None:
        selected_depth_dim = depth_dim or _infer_depth_dim(da)

        if selected_depth_dim is None:
            raise ValueError(
                "depth_index was provided, but no depth dimension could be inferred. "
                "Pass depth_dim explicitly."
            )

        da = da.isel({selected_depth_dim: depth_index})

    if fill_values is not None:
        for value in fill_values:
            da = da.where(da != value)

    if valid_min is not None:
        da = da.where(da >= valid_min)

    if valid_max is not None:
        da = da.where(da <= valid_max)

    return da


def _get_named_array(
    name: str,
    *,
    source: XarrayObject,
    fallback: xr.DataArray,
) -> xr.DataArray:
    """
    Get a coordinate or variable by name from a source object.
    """
    if isinstance(source, xr.Dataset) and name in source:
        return source[name]

    if name in source.coords:
        return source.coords[name]

    if name in fallback.coords:
        return fallback.coords[name]

    raise KeyError(f"Could not find coordinate/variable {name!r}.")


def _resolve_lon_lat(
    *,
    source: XarrayObject,
    field: xr.DataArray,
    lon: str | xr.DataArray | None = None,
    lat: str | xr.DataArray | None = None,
    lon_name: str | None = None,
    lat_name: str | None = None,
) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    """
    Resolve lon/lat arrays for plotting.
    """
    if isinstance(lon, str):
        lon_array = _get_named_array(lon, source=source, fallback=field)
    elif lon is not None:
        lon_array = lon
    elif lon_name is not None:
        lon_array = _get_named_array(lon_name, source=source, fallback=field)
    else:
        lon_array, _ = get_lon_lat(source)

    if isinstance(lat, str):
        lat_array = _get_named_array(lat, source=source, fallback=field)
    elif lat is not None:
        lat_array = lat
    elif lat_name is not None:
        lat_array = _get_named_array(lat_name, source=source, fallback=field)
    else:
        _, lat_array = get_lon_lat(source)

    return lon_array, lat_array


def _save_xarray(
    data: xr.Dataset | xr.DataArray,
    output_path: PathLike,
) -> None:
    """
    Save an xarray Dataset or DataArray to NetCDF.

    If a DataArray has no name, it is temporarily named 'value'.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix != ".nc":
        raise ValueError(
            f"Unsupported output format {suffix!r}. "
            "For now, use NetCDF with extension .nc."
        )

    if isinstance(data, xr.DataArray) and data.name is None:
        data = data.rename("value")

    data.to_netcdf(path)


def _default_group_dim(period: str) -> str | None:
    """
    Return the expected group dimension after period aggregation.
    """
    if period == "seasonal":
        return "season"
    if period == "monthly":
        return "month"
    if period == "annual":
        return "year"
    return None


def run_eda_2d_statistics(
    data: XarrayObject | PathLike | Sequence[PathLike],
    *,
    variable: str | None = None,
    time_dim: str | None = None,
    concat_dim: str | None = None,
    depth_index: int | None = None,
    depth_dim: str | None = None,
    statistics: Sequence[str] = DEFAULT_2D_STATS,
    output_nc: PathLike | None = None,
    output_dir: PathLike | None = None,
    make_plots: bool = True,
    lon: str | xr.DataArray | None = None,
    lat: str | xr.DataArray | None = None,
    lon_name: str | None = None,
    lat_name: str | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    fill_values: Sequence[float] | None = None,
    valid_min: float | None = None,
    valid_max: float | None = None,
    engine: str | None = "netcdf4",
    chunks: dict[str, Any] | None = None,
    show: bool = True,
) -> xr.Dataset:
    """
    Run full-period EDA 2D statistical maps.

    This workflow computes statistics along the time dimension, producing
    gridded fields such as q50, iqr, yki, mean, std, skew, and kurtosis.

    Parameters
    ----------
    data
        xarray object, one NetCDF path, glob pattern, or list of paths.
    variable
        Variable to analyze. If None and data is a Dataset, the first variable is used.
    time_dim
        Time dimension name. Examples: 'time' or 'time_counter'.
    concat_dim
        Dimension used when multiple files are provided.
    depth_index
        Optional vertical level index, for example 0 for surface.
    depth_dim
        Optional depth dimension name.
    statistics
        Statistics to compute.
    output_nc
        Optional NetCDF file path for saving the statistics dataset.
    output_dir
        Optional folder for saving one figure per statistic.
    make_plots
        Whether to create plots when output_dir is provided.
    lon, lat
        Longitude and latitude arrays, or names of variables/coordinates.
    lon_name, lat_name
        Optional longitude/latitude names to find in the source dataset.
    plot_kwargs
        Extra keyword arguments passed to plot_2d_map().
    fill_values
        Values to mask before analysis.
    valid_min, valid_max
        Optional valid range mask.
    engine
        NetCDF backend engine.
    chunks
        Optional dask chunks.
    show
        Whether to display figures.

    Returns
    -------
    xr.Dataset
        Dataset containing one gridded field per statistic.
    """
    opened = _open_input_data(
        data,
        concat_dim=concat_dim,
        engine=engine,
        chunks=chunks,
    )

    da = prepare_2d_field(
        opened,
        variable=variable,
        depth_index=depth_index,
        depth_dim=depth_dim,
        fill_values=fill_values,
        valid_min=valid_min,
        valid_max=valid_max,
    )

    stats_maps = compute_2d_statistics(da, time_dim=time_dim, statistics=statistics)

    if output_nc is not None:
        _save_xarray(stats_maps, output_nc)

    if make_plots and output_dir is not None:
        output_folder = Path(output_dir)
        output_folder.mkdir(parents=True, exist_ok=True)

        lon_array, lat_array = _resolve_lon_lat(
            source=opened,
            field=da,
            lon=lon,
            lat=lat,
            lon_name=lon_name,
            lat_name=lat_name,
        )

        base_name = da.name or "variable"

        for stat_name in stats_maps.data_vars:
            kwargs = dict(plot_kwargs or {})
            kwargs.setdefault("lon", lon_array)
            kwargs.setdefault("lat", lat_array)
            kwargs.setdefault("title", f"{base_name} - {stat_name}")
            kwargs.setdefault("output_path", output_folder / f"{base_name}_{stat_name}.png")
            kwargs.setdefault("show", show)

            plot_2d_map(
                stats_maps[stat_name],
                **kwargs,
            )

    return stats_maps


def run_eda_2d_periods(
    data: XarrayObject | PathLike | Sequence[PathLike],
    *,
    variable: str | None = None,
    time_dim: str | None = None,
    concat_dim: str | None = None,
    depth_index: int | None = None,
    depth_dim: str | None = None,
    periods: Sequence[Period] = ("seasonal", "monthly", "annual"),
    reducer: str = "mean",
    output_dir: PathLike | None = None,
    save_netcdf: bool = False,
    make_plots: bool = True,
    lon: str | xr.DataArray | None = None,
    lat: str | xr.DataArray | None = None,
    lon_name: str | None = None,
    lat_name: str | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    fill_values: Sequence[float] | None = None,
    valid_min: float | None = None,
    valid_max: float | None = None,
    engine: str | None = "netcdf4",
    chunks: dict[str, Any] | None = None,
    show: bool = True,
) -> dict[str, xr.DataArray]:
    """
    Run period-based EDA 2D maps.

    This workflow computes full, seasonal, monthly, and/or annual maps
    using a reducer such as mean, std, median, min, or max.

    Parameters
    ----------
    data
        xarray object, one NetCDF path, glob pattern, or list of paths.
    variable
        Variable to analyze.
    time_dim
        Time dimension name.
    concat_dim
        Dimension used when multiple files are provided.
    depth_index
        Optional vertical level index.
    depth_dim
        Optional depth dimension name.
    periods
        Periods to compute: 'full', 'seasonal', 'monthly', 'annual'.
    reducer
        Aggregation method: 'mean', 'sum', 'std', 'min', 'max', or 'median'.
    output_dir
        Optional output folder for figures and/or NetCDF files.
    save_netcdf
        Whether to save each period result to NetCDF.
    make_plots
        Whether to create period maps.
    lon, lat
        Longitude and latitude arrays, or names of variables/coordinates.
    lon_name, lat_name
        Optional longitude/latitude names to find in the source dataset.
    plot_kwargs
        Extra keyword arguments passed to plotting functions.
    fill_values
        Values to mask before analysis.
    valid_min, valid_max
        Optional valid range mask.
    engine
        NetCDF backend engine.
    chunks
        Optional dask chunks.
    show
        Whether to display figures.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary with one result per requested period.
    """
    opened = _open_input_data(
        data,
        concat_dim=concat_dim,
        engine=engine,
        chunks=chunks,
    )

    da = prepare_2d_field(
        opened,
        variable=variable,
        depth_index=depth_index,
        depth_dim=depth_dim,
        fill_values=fill_values,
        valid_min=valid_min,
        valid_max=valid_max,
    )

    lon_array, lat_array = _resolve_lon_lat(
        source=opened,
        field=da,
        lon=lon,
        lat=lat,
        lon_name=lon_name,
        lat_name=lat_name,
    )

    output_folder = Path(output_dir) if output_dir is not None else None

    if output_folder is not None:
        output_folder.mkdir(parents=True, exist_ok=True)

    base_name = da.name or "variable"
    results: dict[str, xr.DataArray] = {}

    for period in periods:
        period_result = aggregate_periods(
            da,
            period=period,
            time_dim=time_dim,
            reducer=reducer,
        )

        results[period] = period_result

        if output_folder is not None and save_netcdf:
            _save_xarray(
                period_result,
                output_folder / f"{base_name}_{period}_{reducer}.nc",
            )

        if make_plots and output_folder is not None:
            kwargs = dict(plot_kwargs or {})
            kwargs.setdefault("lon", lon_array)
            kwargs.setdefault("lat", lat_array)
            kwargs.setdefault("show", show)

            if period == "full":
                kwargs.setdefault("title", f"{base_name} - full period {reducer}")
                kwargs.setdefault(
                    "output_path",
                    output_folder / f"{base_name}_full_{reducer}.png",
                )

                plot_2d_map(
                    period_result,
                    **kwargs,
                )

            else:
                kwargs.setdefault("group_dim", _default_group_dim(period))
                kwargs.setdefault("title", f"{base_name} - {period} {reducer}")
                kwargs.setdefault(
                    "output_path",
                    output_folder / f"{base_name}_{period}_{reducer}.png",
                )

                plot_2d_subplots(
                    period_result,
                    **kwargs,
                )

    return results


def run_eda_2d(
    data: XarrayObject | PathLike | Sequence[PathLike],
    *,
    variable: str | None = None,
    time_dim: str | None = None,
    concat_dim: str | None = None,
    depth_index: int | None = None,
    depth_dim: str | None = None,
    mode: Mode = "statistics",
    statistics: Sequence[str] = DEFAULT_2D_STATS,
    periods: Sequence[Period] = ("seasonal", "monthly", "annual"),
    reducer: str = "mean",
    output_dir: PathLike | None = None,
    output_statistics_nc: PathLike | None = None,
    save_periods_netcdf: bool = False,
    make_plots: bool = True,
    lon: str | xr.DataArray | None = None,
    lat: str | xr.DataArray | None = None,
    lon_name: str | None = None,
    lat_name: str | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    fill_values: Sequence[float] | None = None,
    valid_min: float | None = None,
    valid_max: float | None = None,
    engine: str | None = "netcdf4",
    chunks: dict[str, Any] | None = None,
    show: bool = True,
) -> xr.Dataset | dict[str, xr.DataArray] | dict[str, object]:
    """
    Run the complete EDA 2D workflow.

    Modes
    -----
    statistics
        Compute full-period statistical maps.
    periods
        Compute full, seasonal, monthly, or annual aggregated maps.
    both
        Run both workflows.

    Returns
    -------
    xr.Dataset, dict, or combined dict
        Depends on mode.
    """
    common_kwargs = {
        "variable": variable,
        "time_dim": time_dim,
        "concat_dim": concat_dim,
        "depth_index": depth_index,
        "depth_dim": depth_dim,
        "make_plots": make_plots,
        "lon": lon,
        "lat": lat,
        "lon_name": lon_name,
        "lat_name": lat_name,
        "plot_kwargs": plot_kwargs,
        "fill_values": fill_values,
        "valid_min": valid_min,
        "valid_max": valid_max,
        "engine": engine,
        "chunks": chunks,
        "show": show,
    }

    if mode == "statistics":
        return run_eda_2d_statistics(
            data,
            statistics=statistics,
            output_nc=output_statistics_nc,
            output_dir=output_dir,
            **common_kwargs,
        )

    if mode == "periods":
        return run_eda_2d_periods(
            data,
            periods=periods,
            reducer=reducer,
            output_dir=output_dir,
            save_netcdf=save_periods_netcdf,
            **common_kwargs,
        )

    if mode == "both":
        stats_output_dir = Path(output_dir) / "statistics" if output_dir is not None else None
        periods_output_dir = Path(output_dir) / "periods" if output_dir is not None else None

        statistics_result = run_eda_2d_statistics(
            data,
            statistics=statistics,
            output_nc=output_statistics_nc,
            output_dir=stats_output_dir,
            **common_kwargs,
        )

        periods_result = run_eda_2d_periods(
            data,
            periods=periods,
            reducer=reducer,
            output_dir=periods_output_dir,
            save_netcdf=save_periods_netcdf,
            **common_kwargs,
        )

        return {
            "statistics": statistics_result,
            "periods": periods_result,
        }

    raise ValueError("mode must be 'statistics', 'periods', or 'both'.")
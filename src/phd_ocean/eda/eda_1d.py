from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .io import (
    expand_paths,
    get_first_data_variable,
    infer_time_dim,
    load_dataset,
    load_many_datasets,
)
from .plotting import plot_violin_by_year

from .stats import (
    DEFAULT_1D_STATS,
    compute_1d_statistics_by_year,
    compute_1d_vector_statistics_by_year,
)

from .vector import compute_speed


PathLike = str | Path
XarrayObject = xr.Dataset | xr.DataArray


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

    Parameters
    ----------
    data
        xarray Dataset or DataArray.
    variable
        Variable name. If None and data is a Dataset, the first data variable is used.

    Returns
    -------
    xr.DataArray
        Selected variable.
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


def prepare_1d_data(
    data: XarrayObject | PathLike | Sequence[PathLike],
    *,
    variable: str | None = None,
    concat_dim: str | None = None,
    depth_index: int | None = None,
    depth_dim: str | None = None,
    fill_values: Sequence[float] | None = None,
    valid_min: float | None = None,
    valid_max: float | None = None,
    engine: str | None = "netcdf4",
    chunks: dict[str, Any] | None = None,
) -> xr.DataArray:
    """
    Prepare data for EDA 1D analysis.

    Steps
    -----
    1. Open input data if a path or list of paths is provided.
    2. Select one variable.
    3. Optionally select one vertical level.
    4. Optionally mask fill values and valid ranges.

    Returns
    -------
    xr.DataArray
        Cleaned DataArray ready for yearly statistics.
    """
    opened = _open_input_data(
        data,
        concat_dim=concat_dim,
        engine=engine,
        chunks=chunks,
    )

    da = select_variable(opened, variable=variable)

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


def xarray_to_yearly_dataframe(
    data: xr.DataArray,
    *,
    time_dim: str | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Convert a DataArray into a DataFrame where each column is one year.

    This replaces the old combination of xarray_to_dataframe()
    and rearranje_dataframe().

    Parameters
    ----------
    data
        Input DataArray.
    time_dim
        Name of time dimension. If None, it is inferred.
    dropna
        Whether to remove NaN/inf values from each yearly column.

    Returns
    -------
    pd.DataFrame
        Columns are years. Rows are samples from that year.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("xarray_to_yearly_dataframe expects an xarray DataArray.")

    current_time_dim = time_dim or infer_time_dim(data)

    if current_time_dim not in data.coords and current_time_dim not in data.dims:
        raise KeyError(f"Time dimension {current_time_dim!r} was not found.")

    years = np.unique(data[current_time_dim].dt.year.values)

    yearly_columns: dict[int, pd.Series] = {}

    for year in years:
        yearly_data = data.where(data[current_time_dim].dt.year == year, drop=True)
        values = np.asarray(yearly_data.values, dtype=float).ravel()

        if dropna:
            values = values[np.isfinite(values)]

        yearly_columns[int(year)] = pd.Series(values)

    yearly_df = pd.DataFrame(yearly_columns)
    yearly_df.columns.name = "year"
    yearly_df.index.name = "sample"

    return yearly_df


def dataframe_to_violin_data(
    yearly_df: pd.DataFrame,
) -> tuple[list[np.ndarray], list[int | str]]:
    """
    Convert a yearly DataFrame into data groups for violin plotting.

    Parameters
    ----------
    yearly_df
        DataFrame with years as columns.

    Returns
    -------
    yearly_data, years
        yearly_data is a list of arrays.
        years is a list of column labels.
    """
    years = yearly_df.columns.tolist()

    yearly_data = [
        yearly_df[year].dropna().to_numpy(dtype=float)
        for year in years
    ]

    return yearly_data, years


def save_summary_table(
    summary: pd.DataFrame,
    output_path: PathLike,
    *,
    float_format: str | None = "%.4f",
) -> None:
    """
    Save a summary table as CSV, Excel, LaTeX, Markdown, or Parquet.

    The file type is inferred from the output extension.

    Supported extensions
    --------------------
    - .csv
    - .xlsx
    - .xls
    - .tex
    - .md
    - .parquet
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        summary.to_csv(path, float_format=float_format)

    elif suffix in {".xlsx", ".xls"}:
        summary.to_excel(path)

    elif suffix == ".tex":
        summary.to_latex(path, float_format=float_format)

    elif suffix == ".md":
        path.write_text(summary.to_markdown(floatfmt=float_format or ""), encoding="utf-8")

    elif suffix == ".parquet":
        summary.to_parquet(path)

    else:
        raise ValueError(
            f"Unsupported table format: {suffix!r}. "
            "Use .csv, .xlsx, .xls, .tex, .md, or .parquet."
        )


def run_eda_1d(
    data: XarrayObject | PathLike | Sequence[PathLike],
    *,
    variable: str | None = None,
    time_dim: str | None = None,
    concat_dim: str | None = None,
    depth_index: int | None = None,
    depth_dim: str | None = None,
    statistics: Sequence[str] = DEFAULT_1D_STATS,
    output_table: PathLike | None = None,
    output_violin: PathLike | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    fill_values: Sequence[float] | None = None,
    valid_min: float | None = None,
    valid_max: float | None = None,
    engine: str | None = "netcdf4",
    chunks: dict[str, Any] | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """
    Run the complete EDA 1D workflow.

    The workflow produces:
    1. A yearly statistical summary table.
    2. Optionally, a yearly violin plot.

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
    output_table
        Optional output table path.
    output_violin
        Optional output figure path.
    ylabel
        Y-axis label for the violin plot.
    title
        Title for the violin plot.
    fill_values
        Values to mask before analysis.
    valid_min, valid_max
        Optional valid range mask.
    engine
        NetCDF backend engine.
    chunks
        Optional dask chunks.
    show
        Whether to display the violin plot.

    Returns
    -------
    pd.DataFrame
        Summary table with statistics as rows and years as columns.
    """
    da = prepare_1d_data(
        data,
        variable=variable,
        concat_dim=concat_dim,
        depth_index=depth_index,
        depth_dim=depth_dim,
        fill_values=fill_values,
        valid_min=valid_min,
        valid_max=valid_max,
        engine=engine,
        chunks=chunks,
    )

    summary = compute_1d_statistics_by_year(
        da,
        time_dim=time_dim,
        statistics=statistics,
    )

    if output_table is not None:
        save_summary_table(summary, output_table)

    if output_violin is not None:
        yearly_df = xarray_to_yearly_dataframe(
            da,
            time_dim=time_dim,
            dropna=True,
        )

        yearly_data, years = dataframe_to_violin_data(yearly_df)

        plot_violin_by_year(
            yearly_data,
            years=years,
            ylabel=ylabel,
            title=title,
            output_path=output_violin,
            show=show,
        )

    return summary

def prepare_1d_vector_data(
    data: XarrayObject | PathLike | Sequence[PathLike],
    *,
    u_variable: str | None = None,
    v_variable: str | None = None,
    v_data: XarrayObject | PathLike | Sequence[PathLike] | None = None,
    concat_dim: str | None = None,
    v_concat_dim: str | None = None,
    depth_index: int | None = None,
    depth_dim: str | None = None,
    fill_values: Sequence[float] | None = None,
    valid_min: float | None = None,
    valid_max: float | None = None,
    engine: str | None = "netcdf4",
    chunks: dict[str, Any] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Prepare paired u/v data for 1D vector EDA.

    This works when:
    1. u and v are in the same Dataset, or
    2. u and v are in separate files/Datasets.

    Parameters
    ----------
    data
        Dataset/DataArray/path for u, or Dataset containing both u and v.
    u_variable
        Name of the u/zonal/eastward component.
    v_variable
        Name of the v/meridional/northward component.
    v_data
        Optional separate Dataset/DataArray/path for v.
    concat_dim
        Concatenation dimension for u data if multiple files are passed.
    v_concat_dim
        Concatenation dimension for v data. If None, concat_dim is used.
    depth_index
        Optional vertical level index.
    depth_dim
        Optional depth dimension name.
    fill_values
        Values to mask.
    valid_min, valid_max
        Optional valid range mask.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Prepared u and v DataArrays.
    """
    if v_data is None:
        opened = _open_input_data(
            data,
            concat_dim=concat_dim,
            engine=engine,
            chunks=chunks,
        )

        u_da = prepare_1d_data(
            opened,
            variable=u_variable,
            depth_index=depth_index,
            depth_dim=depth_dim,
            fill_values=fill_values,
            valid_min=valid_min,
            valid_max=valid_max,
            engine=engine,
            chunks=chunks,
        )

        v_da = prepare_1d_data(
            opened,
            variable=v_variable,
            depth_index=depth_index,
            depth_dim=depth_dim,
            fill_values=fill_values,
            valid_min=valid_min,
            valid_max=valid_max,
            engine=engine,
            chunks=chunks,
        )

    else:
        u_da = prepare_1d_data(
            data,
            variable=u_variable,
            concat_dim=concat_dim,
            depth_index=depth_index,
            depth_dim=depth_dim,
            fill_values=fill_values,
            valid_min=valid_min,
            valid_max=valid_max,
            engine=engine,
            chunks=chunks,
        )

        v_da = prepare_1d_data(
            v_data,
            variable=v_variable,
            concat_dim=v_concat_dim or concat_dim,
            depth_index=depth_index,
            depth_dim=depth_dim,
            fill_values=fill_values,
            valid_min=valid_min,
            valid_max=valid_max,
            engine=engine,
            chunks=chunks,
        )

    u_da, v_da = xr.align(u_da, v_da, join="inner")

    return u_da, v_da


def run_eda_1d_vector(
    data: XarrayObject | PathLike | Sequence[PathLike],
    *,
    u_variable: str | None = None,
    v_variable: str | None = None,
    v_data: XarrayObject | PathLike | Sequence[PathLike] | None = None,
    time_dim: str | None = None,
    concat_dim: str | None = None,
    v_concat_dim: str | None = None,
    depth_index: int | None = None,
    depth_dim: str | None = None,
    direction_convention: str = "bearing",
    direction_units: str = "degrees",
    component_statistics: Sequence[str] = ("mean", "std", "q25", "q50", "q75", "iqr"),
    speed_statistics: Sequence[str] = DEFAULT_1D_STATS,
    output_table: PathLike | None = None,
    output_speed_violin: PathLike | None = None,
    speed_ylabel: str | None = "Speed",
    speed_title: str | None = None,
    fill_values: Sequence[float] | None = None,
    valid_min: float | None = None,
    valid_max: float | None = None,
    engine: str | None = "netcdf4",
    chunks: dict[str, Any] | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """
    Run the complete EDA 1D workflow for paired vector data.

    The workflow produces yearly statistics for:
    - u component
    - v component
    - speed
    - direction using circular statistics

    Optionally, it also creates a yearly speed violin plot.

    Parameters
    ----------
    data
        Dataset/DataArray/path for u, or Dataset containing both u and v.
    u_variable
        Name of the u/zonal/eastward component.
    v_variable
        Name of the v/meridional/northward component.
    v_data
        Optional separate Dataset/DataArray/path for v.
    time_dim
        Time dimension name. Examples: 'time' or 'time_counter'.
    concat_dim
        Dimension used when multiple u files are provided.
    v_concat_dim
        Dimension used when multiple v files are provided.
    depth_index
        Optional vertical level index.
    depth_dim
        Optional depth dimension name.
    direction_convention
        'math', 'bearing', or 'coming_from'.
    direction_units
        'degrees' or 'radians'.
    output_table
        Optional output path for the vector summary table.
    output_speed_violin
        Optional output path for the speed violin plot.

    Returns
    -------
    pd.DataFrame
        Yearly vector statistics table.
    """
    u_da, v_da = prepare_1d_vector_data(
        data,
        u_variable=u_variable,
        v_variable=v_variable,
        v_data=v_data,
        concat_dim=concat_dim,
        v_concat_dim=v_concat_dim,
        depth_index=depth_index,
        depth_dim=depth_dim,
        fill_values=fill_values,
        valid_min=valid_min,
        valid_max=valid_max,
        engine=engine,
        chunks=chunks,
    )

    summary = compute_1d_vector_statistics_by_year(
        u_da,
        v_da,
        time_dim=time_dim,
        direction_convention=direction_convention,
        direction_units=direction_units,
        component_statistics=component_statistics,
        speed_statistics=speed_statistics,
    )

    if output_table is not None:
        save_summary_table(summary, output_table)

    if output_speed_violin is not None:
        speed = compute_speed(u_da, v_da)

        yearly_df = xarray_to_yearly_dataframe(
            speed,
            time_dim=time_dim,
            dropna=True,
        )

        yearly_data, years = dataframe_to_violin_data(yearly_df)

        plot_violin_by_year(
            yearly_data,
            years=years,
            ylabel=speed_ylabel,
            title=speed_title,
            output_path=output_speed_violin,
            show=show,
        )

    return summary
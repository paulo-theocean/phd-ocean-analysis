from __future__ import annotations

from collections.abc import Callable, Sequence
from glob import glob
from pathlib import Path
from typing import Any

import xarray as xr
import numpy as np


PathLike = str | Path
XarrayObject = xr.Dataset | xr.DataArray

TIME_DIM_CANDIDATES = (
    "time",
    "time_counter",
    "Time",
    "TIME",
    "t",
)


def expand_paths(paths: PathLike | Sequence[PathLike]) -> list[Path]:
    """
    Expand one path, one glob pattern, or a list of paths/patterns.

    Examples
    --------
    >>> expand_paths("data/*.nc")
    >>> expand_paths(["data/1992.nc", "data/1993.nc"])
    """
    if isinstance(paths, (str, Path)):
        path_text = str(paths)

        is_pattern = any(char in path_text for char in ["*", "?", "["])
        if is_pattern:
            found = [Path(path) for path in sorted(glob(path_text))]
            if not found:
                raise FileNotFoundError(f"No files matched pattern: {path_text}")
            return found

        path = Path(paths)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        return [path]

    expanded: list[Path] = []
    for item in paths:
        expanded.extend(expand_paths(item))

    # Remove duplicates while preserving order.
    unique_paths: list[Path] = []
    seen: set[str] = set()

    for path in expanded:
        path_key = str(path)
        if path_key not in seen:
            unique_paths.append(path)
            seen.add(path_key)

    return unique_paths


def infer_time_dim(data: XarrayObject) -> str:
    """
    Infer the time dimension or coordinate name.

    This supports datasets that use either 'time' or 'time_counter',
    which appear in your current EDA scripts.
    """
    for name in TIME_DIM_CANDIDATES:
        if name in data.dims or name in data.coords:
            return name

    for dim_name in data.dims:
        if "time" in str(dim_name).lower():
            return str(dim_name)

    for coord_name in data.coords:
        if "time" in str(coord_name).lower():
            return str(coord_name)

        try:
            _ = data[coord_name].dt.year
            return str(coord_name)
        except Exception:
            pass

    raise ValueError(
        "Could not infer the time dimension. "
        "Please pass time_dim explicitly, for example time_dim='time_counter'."
    )


def standardize_time_dim(
    data: XarrayObject,
    *,
    time_dim: str | None = None,
    target: str = "time",
) -> XarrayObject:
    """
    Rename the time dimension/coordinate to a standard name.

    Parameters
    ----------
    data
        xarray Dataset or DataArray.
    time_dim
        Current time dimension name. If None, it is inferred.
    target
        New time dimension name.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Object with standardized time dimension.
    """
    current_time_dim = time_dim or infer_time_dim(data)

    if current_time_dim == target:
        return data

    if target in data.dims or target in data.coords:
        raise ValueError(
            f"Cannot rename {current_time_dim!r} to {target!r} because "
            f"{target!r} already exists in the dataset."
        )

    return data.rename({current_time_dim: target})


def resample_time(
    data: XarrayObject,
    *,
    freq: str,
    time_dim: str | None = None,
    reducer: str = "mean",
) -> XarrayObject:
    """
    Resample data along the time dimension.

    Parameters
    ----------
    data
        xarray Dataset or DataArray.
    freq
        Resampling frequency, for example '7D', '1M', '1YE'.
    time_dim
        Time dimension name. If None, it is inferred.
    reducer
        Aggregation method: 'mean', 'sum', 'std', 'min', or 'max'.
    """
    current_time_dim = time_dim or infer_time_dim(data)
    resampler = data.resample({current_time_dim: freq})

    if reducer == "mean":
        return resampler.mean()
    if reducer == "sum":
        return resampler.sum()
    if reducer == "std":
        return resampler.std()
    if reducer == "min":
        return resampler.min()
    if reducer == "max":
        return resampler.max()

    raise ValueError(
        f"Unsupported reducer: {reducer!r}. "
        "Choose from 'mean', 'sum', 'std', 'min', or 'max'."
    )


def load_dataset(
    file_path: PathLike,
    *,
    engine: str | None = "netcdf4",
    chunks: dict[str, Any] | None = None,
    preprocess: bool = False,
    resample_freq: str = "7D",
    time_dim: str | None = None,
    standardize_time: bool = False,
    target_time_dim: str = "time",
) -> xr.Dataset:
    """
    Load one NetCDF file as an xarray Dataset.

    Parameters
    ----------
    file_path
        Path to the NetCDF file.
    engine
        NetCDF backend engine. Use None to let xarray choose.
    chunks
        Optional dask chunks, for example {'time': 12}.
    preprocess
        Whether to resample the dataset after opening.
    resample_freq
        Resampling frequency used when preprocess=True.
    time_dim
        Time dimension name. If None, it is inferred when needed.
    standardize_time
        Whether to rename the time dimension to target_time_dim.
    target_time_dim
        Target time dimension name, usually 'time'.

    Returns
    -------
    xr.Dataset
        Loaded dataset.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    open_kwargs: dict[str, Any] = {}

    if engine is not None:
        open_kwargs["engine"] = engine

    if chunks is not None:
        open_kwargs["chunks"] = chunks

    dataset = xr.open_dataset(path, **open_kwargs)

    if standardize_time:
        dataset = standardize_time_dim(
            dataset,
            time_dim=time_dim,
            target=target_time_dim,
        )
        time_dim = target_time_dim

    if preprocess:
        dataset = resample_time(
            dataset,
            freq=resample_freq,
            time_dim=time_dim,
            reducer="mean",
        )

    return dataset


def load_many_datasets(
    paths: PathLike | Sequence[PathLike],
    *,
    concat_dim: str,
    transform: Callable[[xr.Dataset], xr.Dataset] | None = None,
    transform_second: Callable[[xr.Dataset], xr.Dataset] | None = None,
    engine: str | None = "netcdf4",
    chunks: dict[str, Any] | None = None,
    load_into_memory: bool = True,
) -> xr.Dataset:
    """
    Load many NetCDF files, optionally transform each one, and concatenate them.

    This replaces the repeated read_netcdfs() function from your EDA scripts.

    Parameters
    ----------
    paths
        One path, one glob pattern, or a sequence of paths/patterns.
    concat_dim
        Dimension used for concatenation, for example 'time' or 'time_counter'.
    transform
        Optional function applied to each dataset before concatenation.
    transform_second
        Optional second transform function.
    engine
        NetCDF backend engine. Use None to let xarray choose.
    chunks
        Optional dask chunks.
    load_into_memory
        If True, load each transformed dataset before closing the file.

    Returns
    -------
    xr.Dataset
        Concatenated dataset.
    """
    file_paths = expand_paths(paths)

    datasets: list[xr.Dataset] = []

    for path in file_paths:
        open_kwargs: dict[str, Any] = {}

        if engine is not None:
            open_kwargs["engine"] = engine

        if chunks is not None:
            open_kwargs["chunks"] = chunks

        with xr.open_dataset(path, **open_kwargs) as dataset:
            processed = dataset

            if transform is not None:
                processed = transform(processed)

            if transform_second is not None:
                processed = transform_second(processed)

            if load_into_memory:
                processed = processed.load()

            datasets.append(processed)

    if not datasets:
        raise ValueError("No datasets were loaded.")

    return xr.concat(datasets, dim=concat_dim)


def transform_longitude(
    data: XarrayObject,
    *,
    lon_name: str = "lon",
    sort: bool = True,
) -> XarrayObject:
    """
    Convert longitudes from 0-360 to -180-180.

    Parameters
    ----------
    data
        xarray Dataset or DataArray.
    lon_name
        Longitude coordinate/variable name.
    sort
        Whether to sort by longitude. Sorting is only applied to 1D longitudes.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Data with transformed longitude.
    """
    if lon_name not in data.coords and lon_name not in data.variables:
        raise KeyError(
            f"Longitude name {lon_name!r} was not found. "
            "Try lon_name='longitude' or lon_name='nav_lon'."
        )

    lon = data[lon_name]
    new_lon = ((lon + 180) % 360) - 180

    transformed = data.assign_coords({lon_name: new_lon})

    if sort and transformed[lon_name].ndim == 1:
        transformed = transformed.sortby(lon_name)

    return transformed


def select_region(
    data: XarrayObject,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    lat_name: str = "lat",
    lon_name: str = "lon",
    drop: bool = True,
) -> XarrayObject:
    """
    Select a rectangular latitude/longitude region.

    This works for regular 1D lat/lon grids and for curvilinear grids
    such as NEMO nav_lat/nav_lon.

    Parameters
    ----------
    data
        xarray Dataset or DataArray.
    lat_min, lat_max
        Latitude bounds.
    lon_min, lon_max
        Longitude bounds.
    lat_name
        Latitude coordinate/variable name.
    lon_name
        Longitude coordinate/variable name.
    drop
        Whether to drop points outside the mask.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Regional subset.
    """
    if lat_name not in data.coords and lat_name not in data.variables:
        raise KeyError(
            f"Latitude name {lat_name!r} was not found. "
            "Try lat_name='latitude' or lat_name='nav_lat'."
        )

    if lon_name not in data.coords and lon_name not in data.variables:
        raise KeyError(
            f"Longitude name {lon_name!r} was not found. "
            "Try lon_name='longitude' or lon_name='nav_lon'."
        )

    lat = data[lat_name]
    lon = data[lon_name]

    # Regular 1D lat/lon grid.
    if lat.ndim == 1 and lon.ndim == 1 and lat_name in data.dims and lon_name in data.dims:
        lat_ascending = bool(lat[0] < lat[-1])
        lon_ascending = bool(lon[0] < lon[-1])

        lat_slice = slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)
        lon_slice = slice(lon_min, lon_max) if lon_ascending else slice(lon_max, lon_min)

        return data.sel({lat_name: lat_slice, lon_name: lon_slice})

    # Curvilinear grid, for example NEMO nav_lat/nav_lon.
    mask = (
        (lat >= lat_min)
        & (lat <= lat_max)
        & (lon >= lon_min)
        & (lon <= lon_max)
    )

    return data.where(mask, drop=drop)


def get_first_data_variable(dataset: xr.Dataset) -> str:
    """
    Return the first data variable name from a Dataset.

    Useful for quick EDA scripts, but for final workflows it is better
    to pass variable explicitly.
    """
    variables = list(dataset.data_vars)

    if not variables:
        raise ValueError("Dataset has no data variables.")

    return variables[0]

def select_vertical_section(
    data: XarrayObject,
    *,
    section_type: str,
    section_value: float,
    lat_name: str = "lat",
    lon_name: str = "lon",
    method: str = "nearest",
) -> XarrayObject:
    """
    Select a vertical section from a gridded dataset.

    Parameters
    ----------
    data
        xarray Dataset or DataArray.
    section_type
        Type of section. Options are:
        - 'latitude' or 'lat': section along a fixed latitude
        - 'longitude' or 'lon': section along a fixed longitude
    section_value
        Latitude or longitude value used to extract the section.
    lat_name
        Name of the latitude coordinate.
    lon_name
        Name of the longitude coordinate.
    method
        Selection method passed to xarray .sel().
        Usually 'nearest' for model grids.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Vertical section.
    """
    valid_section_types = {"latitude", "lat", "longitude", "lon"}

    if section_type not in valid_section_types:
        raise ValueError(
            f"Invalid section_type={section_type!r}. "
            "Use 'latitude', 'lat', 'longitude', or 'lon'."
        )

    if section_type in {"latitude", "lat"}:
        if lat_name not in data.coords and lat_name not in data.dims:
            raise KeyError(f"Latitude coordinate {lat_name!r} was not found.")

        return data.sel({lat_name: section_value}, method=method)

    if lon_name not in data.coords and lon_name not in data.dims:
        raise KeyError(f"Longitude coordinate {lon_name!r} was not found.")

    return data.sel({lon_name: section_value}, method=method)

def select_vertical_section_ij(
    data: XarrayObject,
    *,
    i_indices: Sequence[int],
    j_indices: Sequence[int],
    x_dim: str = "x",
    y_dim: str = "y",
    section_dim: str = "section",
    lon_name: str | None = "nav_lon",
    lat_name: str | None = "nav_lat",
) -> XarrayObject:
    """
    Select a vertical section using model I/J indices.

    This is useful for NEMO-style curvilinear grids where latitude and
    longitude are 2D fields, for example nav_lat/nav_lon.

    Parameters
    ----------
    data
        xarray Dataset or DataArray.
    i_indices
        Sequence of x/I indices along the section.
    j_indices
        Sequence of y/J indices along the section.
    x_dim
        Name of the model x dimension.
    y_dim
        Name of the model y dimension.
    section_dim
        Name of the new section dimension.
    lon_name
        Optional longitude variable name to attach along the section.
    lat_name
        Optional latitude variable name to attach along the section.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Data selected along the I/J section.

    Examples
    --------
    >>> section = select_vertical_section_ij(
    ...     ds,
    ...     i_indices=[10, 11, 12, 13],
    ...     j_indices=[40, 40, 40, 40],
    ...     x_dim="x",
    ...     y_dim="y",
    ... )
    """
    i_array = np.asarray(i_indices, dtype=int)
    j_array = np.asarray(j_indices, dtype=int)

    if i_array.ndim != 1 or j_array.ndim != 1:
        raise ValueError("i_indices and j_indices must be 1D sequences.")

    if len(i_array) != len(j_array):
        raise ValueError(
            "i_indices and j_indices must have the same length. "
            f"Got {len(i_array)} I values and {len(j_array)} J values."
        )

    if x_dim not in data.dims:
        raise KeyError(
            f"x_dim={x_dim!r} was not found in data dimensions: {list(data.dims)}"
        )

    if y_dim not in data.dims:
        raise KeyError(
            f"y_dim={y_dim!r} was not found in data dimensions: {list(data.dims)}"
        )

    i_indexer = xr.DataArray(
        i_array,
        dims=section_dim,
        name="i",
    )

    j_indexer = xr.DataArray(
        j_array,
        dims=section_dim,
        name="j",
    )

    section = data.isel({x_dim: i_indexer, y_dim: j_indexer})

    section = section.assign_coords(
        {
            "i": (section_dim, i_array),
            "j": (section_dim, j_array),
        }
    )

    # Attach lon/lat along the section when available.
    if lon_name is not None and lon_name in data:
        try:
            section_lon = data[lon_name].isel({x_dim: i_indexer, y_dim: j_indexer})
            section = section.assign_coords({lon_name: section_lon})
        except Exception:
            pass

    if lat_name is not None and lat_name in data:
        try:
            section_lat = data[lat_name].isel({x_dim: i_indexer, y_dim: j_indexer})
            section = section.assign_coords({lat_name: section_lat})
        except Exception:
            pass

    return section

def make_section_indices(
    *,
    i_start: int,
    i_end: int,
    j_start: int,
    j_end: int,
    n_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create I/J indices for a straight section between two model-grid points.

    Parameters
    ----------
    i_start, i_end
        Start and end I indices.
    j_start, j_end
        Start and end J indices.
    n_points
        Number of section points. If None, it is estimated from the
        largest index difference.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        I indices and J indices.
    """
    if n_points is None:
        n_points = max(abs(i_end - i_start), abs(j_end - j_start)) + 1

    i_indices = np.rint(np.linspace(i_start, i_end, n_points)).astype(int)
    j_indices = np.rint(np.linspace(j_start, j_end, n_points)).astype(int)

    # Remove repeated points caused by rounding.
    points = list(dict.fromkeys(zip(i_indices, j_indices)))

    i_unique = np.array([point[0] for point in points], dtype=int)
    j_unique = np.array([point[1] for point in points], dtype=int)

    return i_unique, j_unique
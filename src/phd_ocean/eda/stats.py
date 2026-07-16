from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as scipy_stats
from collections.abc import Sequence

from .io import infer_time_dim


XarrayObject = xr.Dataset | xr.DataArray

DEFAULT_1D_STATS = (
    "mean",
    "median",
    "q25",
    "q75",
    "iqr",
    "variance",
    "std",
    "mad",
    "skew",
    "yki",
    "kurtosis",
)

DEFAULT_2D_STATS = (
    "q25",
    "q50",
    "q75",
    "iqr",
    "yki",
    "mad",
    "std",
    "mean",
    "skew",
    "kurtosis",
)

Reducer = Literal["mean", "sum", "std", "min", "max", "median"]


def _normalize_stat_name(name: str) -> str:
    """
    Normalize common statistic aliases.
    """
    aliases = {
        "q1": "q25",
        "p25": "q25",
        "percentile25": "q25",
        "median": "q50",
        "q2": "q50",
        "p50": "q50",
        "percentile50": "q50",
        "q3": "q75",
        "p75": "q75",
        "percentile75": "q75",
        "std_dev": "std",
        "standard_deviation": "std",
        "var": "variance",
        "skw": "skew",
        "kurt": "kurtosis",
        "kurtosis_fisher": "kurtosis",
        "circ_mean": "circular_mean",
        "circularmean": "circular_mean",
        "circ_std": "circular_std",
        "circularstd": "circular_std",
        "circ_var": "circular_variance",
        "circularvar": "circular_variance"
    }

    clean_name = name.lower().strip()
    return aliases.get(clean_name, clean_name)


def _as_clean_1d_array(data) -> np.ndarray:
    """
    Convert input data to a 1D NumPy array and remove NaN values.
    """
    values = np.asarray(data, dtype=float).ravel()
    return values[np.isfinite(values)]


def _safe_divide(numerator, denominator):
    """
    Divide while avoiding infinite values when denominator is zero.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator

    if np.isscalar(result):
        return result if np.isfinite(result) else np.nan

    return xr.where(np.isfinite(result), result, np.nan)


def compute_1d_statistics(
    data,
    *,
    statistics: Sequence[str] = DEFAULT_1D_STATS,
) -> pd.Series:
    """
    Compute statistical summaries for one 1D array.

    Parameters
    ----------
    data
        Array-like data. NaN values are ignored.
    statistics
        Statistics to compute.

    Returns
    -------
    pd.Series
        One statistic per row.
    """
    values = _as_clean_1d_array(data)

    normalized_stats = [_normalize_stat_name(stat) for stat in statistics]
    output: dict[str, float] = {}

    if values.size == 0:
        return pd.Series(
            {stat: np.nan for stat in normalized_stats},
            name="value",
            dtype=float,
        )

    q25 = np.nanpercentile(values, 25)
    q50 = np.nanpercentile(values, 50)
    q75 = np.nanpercentile(values, 75)
    iqr = q75 - q25

    for stat_name in normalized_stats:
        if stat_name == "mean":
            output["mean"] = float(np.nanmean(values))

        elif stat_name == "q50":
            output["median"] = float(q50)

        elif stat_name == "q25":
            output["q25"] = float(q25)

        elif stat_name == "q75":
            output["q75"] = float(q75)

        elif stat_name == "iqr":
            output["iqr"] = float(iqr)

        elif stat_name == "variance":
            output["variance"] = float(np.nanvar(values))

        elif stat_name == "std":
            output["std"] = float(np.nanstd(values))

        elif stat_name == "mad":
            output["mad"] = float(scipy_stats.median_abs_deviation(values, nan_policy="omit"))

        elif stat_name == "skew":
            output["skew"] = float(scipy_stats.skew(values, nan_policy="omit"))

        elif stat_name == "yki":
            output["yki"] = float(_safe_divide(q25 - 2 * q50 + q75, iqr))

        elif stat_name == "kurtosis":
            output["kurtosis"] = float(
                scipy_stats.kurtosis(values, nan_policy="omit", fisher=True)
            )

        elif stat_name == "min":
            output["min"] = float(np.nanmin(values))

        elif stat_name == "max":
            output["max"] = float(np.nanmax(values))

        elif stat_name == "count":
            output["count"] = float(values.size)

        else:
            raise ValueError(f"Unsupported statistic: {stat_name!r}")

    return pd.Series(output, name="value")


def compute_1d_statistics_by_year(
    data: xr.DataArray,
    *,
    time_dim: str | None = None,
    statistics: Sequence[str] = DEFAULT_1D_STATS,
) -> pd.DataFrame:
    """
    Compute yearly statistical summaries from a DataArray.

    This is the clean replacement for the repeated yearly summary logic
    in your EDA 1D scripts.

    Parameters
    ----------
    data
        Input DataArray.
    time_dim
        Time dimension name. If None, it is inferred.
    statistics
        Statistics to compute.

    Returns
    -------
    pd.DataFrame
        Rows are statistics and columns are years.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("compute_1d_statistics_by_year expects an xarray DataArray.")

    current_time_dim = time_dim or infer_time_dim(data)

    if current_time_dim not in data.coords and current_time_dim not in data.dims:
        raise KeyError(f"Time dimension {current_time_dim!r} was not found.")

    years = np.unique(data[current_time_dim].dt.year.values)

    yearly_results: dict[int, pd.Series] = {}

    for year in years:
        yearly_data = data.where(data[current_time_dim].dt.year == year, drop=True)
        yearly_results[int(year)] = compute_1d_statistics(
            yearly_data.values,
            statistics=statistics,
        )

    summary = pd.DataFrame(yearly_results)
    summary.columns.name = "year"
    summary.index.name = "statistic"

    return summary


def _apply_nanpercentile(
    data: xr.DataArray,
    *,
    q: float,
    time_dim: str,
) -> xr.DataArray:
    """
    Apply np.nanpercentile along one xarray time dimension.
    """
    return xr.apply_ufunc(
        np.nanpercentile,
        data,
        input_core_dims=[[time_dim]],
        kwargs={"q": q, "axis": -1},
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )


def _apply_scipy_stat(
    function,
    data: xr.DataArray,
    *,
    time_dim: str,
    **kwargs,
) -> xr.DataArray:
    """
    Apply a scipy statistic along one xarray time dimension.
    """
    return xr.apply_ufunc(
        function,
        data,
        input_core_dims=[[time_dim]],
        kwargs={"axis": -1, "nan_policy": "omit", **kwargs},
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )


def compute_2d_statistics(
    data: xr.DataArray,
    *,
    time_dim: str | None = None,
    statistics: Sequence[str] = DEFAULT_2D_STATS,
    circular_units: str = "degrees",
) -> xr.Dataset:
    """
    Compute gridded statistics along the time dimension.

    This is the clean replacement for your StatisticalMoments2D workflow.

    Parameters
    ----------
    data
        Input DataArray with dimensions like time/y/x or time/depth/y/x.
    time_dim
        Time dimension name. If None, it is inferred.
    statistics
        Statistics to compute.

    Returns
    -------
    xr.Dataset
        One variable per statistic.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("compute_2d_statistics expects an xarray DataArray.")

    current_time_dim = time_dim or infer_time_dim(data)

    if current_time_dim not in data.dims:
        raise KeyError(
            f"Time dimension {current_time_dim!r} was not found in data dimensions: "
            f"{list(data.dims)}"
        )

    requested_stats = [_normalize_stat_name(stat) for stat in statistics]
    results: dict[str, xr.DataArray] = {}

    q25 = None
    q50 = None
    q75 = None

    need_q25 = any(stat in requested_stats for stat in ("q25", "iqr", "yki"))
    need_q50 = any(stat in requested_stats for stat in ("q50", "mad", "yki"))
    need_q75 = any(stat in requested_stats for stat in ("q75", "iqr", "yki"))

    if need_q25:
        q25 = _apply_nanpercentile(data, q=25, time_dim=current_time_dim)
        q25.name = "q25"

    if need_q50:
        q50 = _apply_nanpercentile(data, q=50, time_dim=current_time_dim)
        q50.name = "q50"

    if need_q75:
        q75 = _apply_nanpercentile(data, q=75, time_dim=current_time_dim)
        q75.name = "q75"

    for stat_name in requested_stats:
        if stat_name == "q25":
            results["q25"] = q25

        elif stat_name == "q50":
            results["q50"] = q50

        elif stat_name == "q75":
            results["q75"] = q75

        elif stat_name == "iqr":
            iqr = q75 - q25
            iqr.name = "iqr"
            results["iqr"] = iqr

        elif stat_name == "yki":
            iqr = q75 - q25
            yki = _safe_divide(q25 - 2 * q50 + q75, iqr)
            yki.name = "yki"
            results["yki"] = yki

        elif stat_name == "mad":
            mad = np.abs(data - q50).median(dim=current_time_dim, skipna=True)
            mad.name = "mad"
            results["mad"] = mad

        elif stat_name == "std":
            std = data.std(dim=current_time_dim, skipna=True)
            std.name = "std"
            results["std"] = std

        elif stat_name == "mean":
            mean = data.mean(dim=current_time_dim, skipna=True)
            mean.name = "mean"
            results["mean"] = mean

        elif stat_name == "variance":
            variance = data.var(dim=current_time_dim, skipna=True)
            variance.name = "variance"
            results["variance"] = variance

        elif stat_name == "skew":
            skew = _apply_scipy_stat(
                scipy_stats.skew,
                data,
                time_dim=current_time_dim,
            )
            skew.name = "skew"
            results["skew"] = skew

        elif stat_name == "kurtosis":
            kurtosis = _apply_scipy_stat(
                scipy_stats.kurtosis,
                data,
                time_dim=current_time_dim,
                fisher=True,
            )
            kurtosis.name = "kurtosis"
            results["kurtosis"] = kurtosis

        elif stat_name == "min":
            minimum = data.min(dim=current_time_dim, skipna=True)
            minimum.name = "min"
            results["min"] = minimum

        elif stat_name == "max":
            maximum = data.max(dim=current_time_dim, skipna=True)
            maximum.name = "max"
            results["max"] = maximum
        elif stat_name == "circular_mean":
            circular_mean = circular_mean_direction(
                data,
                dim=current_time_dim,
                units=circular_units,
            )
            results["circular_mean"] = circular_mean

        elif stat_name == "circular_std":
            circular_std = circular_std_direction(
                data,
                dim=current_time_dim,
                units=circular_units,
            )
            results["circular_std"] = circular_std

        elif stat_name == "circular_variance":
            circular_variance = circular_variance_direction(
                data,
                dim=current_time_dim,
                units=circular_units,
            )
            results["circular_variance"] = circular_variance
        else:
            raise ValueError(f"Unsupported 2D statistic: {stat_name!r}")

    return xr.Dataset(results)


def _reduce_data(
    data: XarrayObject,
    *,
    dim: str,
    reducer: Reducer,
) -> XarrayObject:
    """
    Reduce xarray data along one dimension.
    """
    if reducer == "mean":
        return data.mean(dim=dim, skipna=True)
    if reducer == "sum":
        return data.sum(dim=dim, skipna=True)
    if reducer == "std":
        return data.std(dim=dim, skipna=True)
    if reducer == "min":
        return data.min(dim=dim, skipna=True)
    if reducer == "max":
        return data.max(dim=dim, skipna=True)
    if reducer == "median":
        return data.median(dim=dim, skipna=True)

    raise ValueError(
        f"Unsupported reducer: {reducer!r}. "
        "Choose from 'mean', 'sum', 'std', 'min', 'max', or 'median'."
    )


def _reduce_groupby(grouped, *, reducer: Reducer):
    """
    Reduce an xarray groupby object.
    """
    if reducer == "mean":
        return grouped.mean(skipna=True)
    if reducer == "sum":
        return grouped.sum(skipna=True)
    if reducer == "std":
        return grouped.std(skipna=True)
    if reducer == "min":
        return grouped.min(skipna=True)
    if reducer == "max":
        return grouped.max(skipna=True)
    if reducer == "median":
        return grouped.median(skipna=True)

    raise ValueError(
        f"Unsupported reducer: {reducer!r}. "
        "Choose from 'mean', 'sum', 'std', 'min', 'max', or 'median'."
    )


def _month_to_season(month: int, seasons: dict[str, list[int]]) -> str:
    """
    Convert month number to a custom season label.
    """
    for season_name, months in seasons.items():
        if int(month) in months:
            return season_name

    raise ValueError(f"Month {month!r} was not assigned to any season.")


def aggregate_periods(
    data: XarrayObject,
    *,
    period: Literal["full", "seasonal", "monthly", "annual"],
    time_dim: str | None = None,
    reducer: Reducer = "mean",
    seasons: dict[str, list[int]] | None = None,
    season_coord: str = "season",
) -> XarrayObject:
    """
    Aggregate data by full period, season, month, or year.

    Parameters
    ----------
    data
        xarray Dataset or DataArray.
    period
        Aggregation period:
        - 'full'
        - 'seasonal'
        - 'monthly'
        - 'annual'
    time_dim
        Time dimension name. If None, it is inferred.
    reducer
        Aggregation method.
    seasons
        Custom season dictionary. Example:
        {'JFM': [1, 2, 3], 'AMJ': [4, 5, 6], ...}
    season_coord
        Name of the season coordinate.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Aggregated data.
    """
    current_time_dim = time_dim or infer_time_dim(data)

    if current_time_dim not in data.dims and current_time_dim not in data.coords:
        raise KeyError(f"Time dimension {current_time_dim!r} was not found.")

    if seasons is None:
        seasons = {
            "JFM": [1, 2, 3],
            "AMJ": [4, 5, 6],
            "JAS": [7, 8, 9],
            "OND": [10, 11, 12],
        }

    if period == "full":
        return _reduce_data(data, dim=current_time_dim, reducer=reducer)

    if period == "monthly":
        return _reduce_groupby(
            data.groupby(f"{current_time_dim}.month"),
            reducer=reducer,
        )

    if period == "annual":
        return _reduce_groupby(
            data.groupby(f"{current_time_dim}.year"),
            reducer=reducer,
        )

    if period == "seasonal":
        months = data[current_time_dim].dt.month

        season_labels = xr.apply_ufunc(
            lambda month: _month_to_season(month, seasons),
            months,
            vectorize=True,
            dask="allowed",
            output_dtypes=[str],
        )

        data_with_season = data.assign_coords({season_coord: season_labels})

        result = _reduce_groupby(
            data_with_season.groupby(season_coord),
            reducer=reducer,
        )

        # Preserve the user-defined season order.
        ordered_seasons = [
            season_name
            for season_name in seasons
            if season_name in result[season_coord].values
        ]

        return result.sel({season_coord: ordered_seasons})

    raise ValueError(
        f"Unsupported period: {period!r}. "
        "Choose from 'full', 'seasonal', 'monthly', or 'annual'."
    )

def _circular_bounds(
    *,
    units: str = "degrees",
    low: float | None = None,
    high: float | None = None,
) -> tuple[float, float]:
    """
    Return circular-statistic bounds.

    For degrees, default is 0 to 360.
    For radians, default is 0 to 2*pi.
    """
    if low is not None and high is not None:
        return low, high

    if units == "degrees":
        return 0.0, 360.0

    if units == "radians":
        return 0.0, float(2 * np.pi)

    raise ValueError("units must be 'degrees' or 'radians'.")


def circular_mean_direction(
    data: xr.DataArray,
    *,
    dim: str | None = None,
    units: str = "degrees",
    low: float | None = None,
    high: float | None = None,
) -> xr.DataArray:
    """
    Compute circular mean direction.

    This should be used for direction fields instead of normal arithmetic mean.
    """
    current_dim = dim or infer_time_dim(data)
    low_value, high_value = _circular_bounds(
        units=units,
        low=low,
        high=high,
    )

    result = xr.apply_ufunc(
        scipy_stats.circmean,
        data,
        input_core_dims=[[current_dim]],
        kwargs={
            "low": low_value,
            "high": high_value,
            "axis": -1,
            "nan_policy": "omit",
        },
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )

    result.name = "circular_mean"
    result.attrs.update(
        {
            "long_name": "circular mean direction",
            "units": units,
        }
    )

    return result


def circular_std_direction(
    data: xr.DataArray,
    *,
    dim: str | None = None,
    units: str = "degrees",
    low: float | None = None,
    high: float | None = None,
) -> xr.DataArray:
    """
    Compute circular standard deviation of direction.
    """
    current_dim = dim or infer_time_dim(data)
    low_value, high_value = _circular_bounds(
        units=units,
        low=low,
        high=high,
    )

    result = xr.apply_ufunc(
        scipy_stats.circstd,
        data,
        input_core_dims=[[current_dim]],
        kwargs={
            "low": low_value,
            "high": high_value,
            "axis": -1,
            "nan_policy": "omit",
        },
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )

    result.name = "circular_std"
    result.attrs.update(
        {
            "long_name": "circular standard deviation of direction",
            "units": units,
        }
    )

    return result


def circular_variance_direction(
    data: xr.DataArray,
    *,
    dim: str | None = None,
    units: str = "degrees",
    low: float | None = None,
    high: float | None = None,
) -> xr.DataArray:
    """
    Compute circular variance of direction.
    """
    current_dim = dim or infer_time_dim(data)
    low_value, high_value = _circular_bounds(
        units=units,
        low=low,
        high=high,
    )

    result = xr.apply_ufunc(
        scipy_stats.circvar,
        data,
        input_core_dims=[[current_dim]],
        kwargs={
            "low": low_value,
            "high": high_value,
            "axis": -1,
            "nan_policy": "omit",
        },
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )

    result.name = "circular_variance"
    result.attrs.update(
        {
            "long_name": "circular variance of direction",
            "units": units,
        }
    )

    return result

def compute_1d_circular_statistics(
    data,
    *,
    units: str = "degrees",
    low: float | None = None,
    high: float | None = None,
) -> pd.Series:
    """
    Compute 1D circular statistics for direction data.

    Use this for directions/angles instead of normal arithmetic statistics.
    """
    values = _as_clean_1d_array(data)

    low_value, high_value = _circular_bounds(
        units=units,
        low=low,
        high=high,
    )

    if values.size == 0:
        return pd.Series(
            {
                "circular_mean": np.nan,
                "circular_std": np.nan,
                "circular_variance": np.nan,
                "resultant_length": np.nan,
                "count": 0.0,
            },
            dtype=float,
        )

    period = high_value - low_value
    angles = (values - low_value) * 2 * np.pi / period

    sin_mean = np.nanmean(np.sin(angles))
    cos_mean = np.nanmean(np.cos(angles))

    resultant_length = np.sqrt(sin_mean**2 + cos_mean**2)
    resultant_length = np.clip(resultant_length, 0.0, 1.0)

    mean_angle = np.arctan2(sin_mean, cos_mean) % (2 * np.pi)
    circular_mean = mean_angle * period / (2 * np.pi) + low_value

    circular_variance = 1.0 - resultant_length

    if resultant_length <= 0:
        circular_std = np.nan
    else:
        circular_std = np.sqrt(-2 * np.log(resultant_length)) * period / (2 * np.pi)

    return pd.Series(
        {
            "circular_mean": float(circular_mean),
            "circular_std": float(circular_std),
            "circular_variance": float(circular_variance),
            "resultant_length": float(resultant_length),
            "count": float(values.size),
        },
        name="value",
    )


def compute_1d_vector_statistics(
    u,
    v,
    *,
    direction_convention: str = "bearing",
    direction_units: str = "degrees",
    component_statistics: Sequence[str] = ("mean", "std", "q25", "q50", "q75", "iqr"),
    speed_statistics: Sequence[str] = DEFAULT_1D_STATS,
) -> pd.Series:
    """
    Compute 1D statistics for paired vector components.

    Outputs include:
    - u component statistics
    - v component statistics
    - speed statistics
    - direction circular statistics
    - mean vector magnitude and direction
    """
    u_values = np.asarray(u, dtype=float).ravel()
    v_values = np.asarray(v, dtype=float).ravel()

    valid = np.isfinite(u_values) & np.isfinite(v_values)

    u_values = u_values[valid]
    v_values = v_values[valid]

    if u_values.size == 0:
        return pd.Series(dtype=float)

    speed = np.sqrt(u_values**2 + v_values**2)

    if direction_convention == "math":
        direction_rad = np.arctan2(v_values, u_values)

    elif direction_convention == "bearing":
        direction_rad = np.arctan2(u_values, v_values)

    elif direction_convention == "coming_from":
        direction_rad = np.arctan2(u_values, v_values) + np.pi

    else:
        raise ValueError(
            "direction_convention must be 'math', 'bearing', or 'coming_from'."
        )

    if direction_units == "degrees":
        direction = (np.degrees(direction_rad) + 360) % 360

    elif direction_units == "radians":
        direction = direction_rad % (2 * np.pi)

    else:
        raise ValueError("direction_units must be 'degrees' or 'radians'.")

    u_stats = compute_1d_statistics(
        u_values,
        statistics=component_statistics,
    ).add_prefix("u_")

    v_stats = compute_1d_statistics(
        v_values,
        statistics=component_statistics,
    ).add_prefix("v_")

    speed_stats = compute_1d_statistics(
        speed,
        statistics=speed_statistics,
    ).add_prefix("speed_")

    direction_stats = compute_1d_circular_statistics(
        direction,
        units=direction_units,
    ).add_prefix("direction_")

    mean_u = np.nanmean(u_values)
    mean_v = np.nanmean(v_values)

    mean_vector_speed = np.sqrt(mean_u**2 + mean_v**2)

    if direction_convention == "math":
        mean_direction_rad = np.arctan2(mean_v, mean_u)

    elif direction_convention == "bearing":
        mean_direction_rad = np.arctan2(mean_u, mean_v)

    else:
        mean_direction_rad = np.arctan2(mean_u, mean_v) + np.pi

    if direction_units == "degrees":
        mean_vector_direction = (np.degrees(mean_direction_rad) + 360) % 360
    else:
        mean_vector_direction = mean_direction_rad % (2 * np.pi)

    vector_summary = pd.Series(
        {
            "mean_vector_u": float(mean_u),
            "mean_vector_v": float(mean_v),
            "mean_vector_speed": float(mean_vector_speed),
            "mean_vector_direction": float(mean_vector_direction),
            "count": float(u_values.size),
        }
    )

    return pd.concat(
        [
            vector_summary,
            u_stats,
            v_stats,
            speed_stats,
            direction_stats,
        ]
    )


def compute_1d_vector_statistics_by_year(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    time_dim: str | None = None,
    direction_convention: str = "bearing",
    direction_units: str = "degrees",
    component_statistics: Sequence[str] = ("mean", "std", "q25", "q50", "q75", "iqr"),
    speed_statistics: Sequence[str] = DEFAULT_1D_STATS,
) -> pd.DataFrame:
    """
    Compute yearly 1D vector statistics from paired u/v DataArrays.

    Returns
    -------
    pd.DataFrame
        Rows are vector statistics and columns are years.
    """
    if not isinstance(u, xr.DataArray):
        raise TypeError("u must be an xarray DataArray.")

    if not isinstance(v, xr.DataArray):
        raise TypeError("v must be an xarray DataArray.")

    current_time_dim = time_dim or infer_time_dim(u)

    u_aligned, v_aligned = xr.align(u, v, join="inner")

    years = np.unique(u_aligned[current_time_dim].dt.year.values)

    yearly_results: dict[int, pd.Series] = {}

    for year in years:
        year_mask = u_aligned[current_time_dim].dt.year == year

        u_year = u_aligned.where(year_mask, drop=True)
        v_year = v_aligned.where(year_mask, drop=True)

        yearly_results[int(year)] = compute_1d_vector_statistics(
            u_year.values,
            v_year.values,
            direction_convention=direction_convention,
            direction_units=direction_units,
            component_statistics=component_statistics,
            speed_statistics=speed_statistics,
        )

    summary = pd.DataFrame(yearly_results)
    summary.columns.name = "year"
    summary.index.name = "statistic"

    return summary
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker


PathLike = str | Path
XarrayObject = xr.Dataset | xr.DataArray


LON_NAME_CANDIDATES = (
    "lon",
    "longitude",
    "nav_lon",
    "nav_lon_grid_T",
    "nav_lon_grid_U",
    "nav_lon_grid_V",
)

LAT_NAME_CANDIDATES = (
    "lat",
    "latitude",
    "nav_lat",
    "nav_lat_grid_T",
    "nav_lat_grid_U",
    "nav_lat_grid_V",
)


def _save_or_show(
    fig: plt.Figure,
    *,
    output_path: PathLike | None = None,
    show: bool = True,
    dpi: int = 300,
    transparent: bool = False,
) -> None:
    """
    Save and/or show a matplotlib figure.
    """
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output,
            dpi=dpi,
            transparent=transparent,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def _split_map_feature_kwargs(
    map_features: dict[str, Any] | None,
    *,
    default_draw_labels: bool = False,
    default_extent_crs: ccrs.Projection = ccrs.PlateCarree(),
    default_gridline_crs: ccrs.Projection = ccrs.PlateCarree(),
) -> tuple[bool, ccrs.Projection, ccrs.Projection, dict[str, Any]]:
    """
    Split special map-feature keywords from the rest.

    This prevents errors like:

        TypeError: got multiple values for keyword argument 'draw_labels'

    when the user passes:

        map_features={"draw_labels": False}

    and the plotting function also passes draw_labels explicitly.
    """
    feature_kwargs = dict(map_features or {})

    draw_labels = feature_kwargs.pop("draw_labels", default_draw_labels)
    extent_crs = feature_kwargs.pop("extent_crs", default_extent_crs)
    gridline_crs = feature_kwargs.pop("gridline_crs", default_gridline_crs)

    return draw_labels, extent_crs, gridline_crs, feature_kwargs


def get_lon_lat(
    data: XarrayObject,
    *,
    lon_names: Sequence[str] = LON_NAME_CANDIDATES,
    lat_names: Sequence[str] = LAT_NAME_CANDIDATES,
) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    """
    Try to detect longitude and latitude arrays from common names.

    Supports regular lon/lat grids and NEMO-style nav_lon/nav_lat grids.
    """
    lon = None
    lat = None

    for name in lon_names:
        if name in data.coords or name in data.variables:
            lon = data[name]
            break

    for name in lat_names:
        if name in data.coords or name in data.variables:
            lat = data[name]
            break

    return lon, lat


def _dataarray_to_plot_values(
    da: xr.DataArray,
) -> np.ndarray:
    """
    Convert a DataArray to a squeezed 2D NumPy array for plotting.
    """
    values = np.asarray(da.squeeze().values)

    if values.ndim != 2:
        raise ValueError(
            "The field to plot must be 2D after squeezing. "
            f"Got shape {values.shape}."
        )

    return values


def _prepare_lon_lat_mesh(
    data: XarrayObject,
    *,
    lon: xr.DataArray | np.ndarray | None = None,
    lat: xr.DataArray | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare longitude and latitude arrays for contourf/pcolormesh.
    """
    if lon is None or lat is None:
        detected_lon, detected_lat = get_lon_lat(data)
        lon = detected_lon if lon is None else lon
        lat = detected_lat if lat is None else lat

    if lon is None or lat is None:
        raise ValueError(
            "Could not find lon/lat coordinates. "
            "Pass lon=... and lat=... explicitly."
        )

    lon_values = np.asarray(lon)
    lat_values = np.asarray(lat)

    if lon_values.ndim == 1 and lat_values.ndim == 1:
        lon_values, lat_values = np.meshgrid(lon_values, lat_values)

    return lon_values, lat_values


def _thin_2d_field(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    stride: int | None = None,
    max_vectors: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Thin 2D vector fields for quiver plotting.
    """
    if x.ndim != 2 or y.ndim != 2 or u.ndim != 2 or v.ndim != 2:
        raise ValueError(
            "x, y, u, and v must all be 2D arrays. "
            f"Got shapes x={x.shape}, y={y.shape}, u={u.shape}, v={v.shape}."
        )

    if stride is None:
        ny, nx = u.shape
        stride_y = max(1, int(np.ceil(ny / max_vectors)))
        stride_x = max(1, int(np.ceil(nx / max_vectors)))
    else:
        stride_y = stride
        stride_x = stride

    return (
        x[::stride_y, ::stride_x],
        y[::stride_y, ::stride_x],
        u[::stride_y, ::stride_x],
        v[::stride_y, ::stride_x],
    )


def plot_quiver(
    ax: plt.Axes,
    u: xr.DataArray | np.ndarray,
    v: xr.DataArray | np.ndarray,
    *,
    lon: xr.DataArray | np.ndarray,
    lat: xr.DataArray | np.ndarray,
    stride: int | None = None,
    max_vectors: int = 25,
    scale: float | None = None,
    width: float | None = None,
    data_crs: ccrs.Projection = ccrs.PlateCarree(),
    quiver_kwargs: dict[str, Any] | None = None,
):
    """
    Add vector arrows to an existing map axis.
    """
    x = np.asarray(lon)
    y = np.asarray(lat)

    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)

    u_values = np.squeeze(np.asarray(u))
    v_values = np.squeeze(np.asarray(v))

    x_thin, y_thin, u_thin, v_thin = _thin_2d_field(
        x,
        y,
        u_values,
        v_values,
        stride=stride,
        max_vectors=max_vectors,
    )

    kwargs = dict(quiver_kwargs or {})

    if scale is not None:
        kwargs["scale"] = scale

    if width is not None:
        kwargs["width"] = width

    return ax.quiver(
        x_thin,
        y_thin,
        u_thin,
        v_thin,
        transform=data_crs,
        **kwargs,
    )


def add_map_features(
    ax: plt.Axes,
    *,
    extent: Sequence[float] | None = None,
    extent_crs: ccrs.Projection = ccrs.PlateCarree(),
    gridline_crs: ccrs.Projection = ccrs.PlateCarree(),
    gridline_step: int | float | None = None,
    draw_labels: bool = True,
    land: bool = True,
    coastline: bool = True,
    borders: bool = False,
    states: bool = False,
    gridline_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Add Cartopy map features and lon/lat gridlines.

    The map projection can be Albers, Lambert, PlateCarree, etc.
    The extent and gridlines are interpreted in lon/lat coordinates by default.
    """
    if extent is not None:
        ax.set_extent(extent, crs=extent_crs)

    if land:
        ax.add_feature(
            cfeature.LAND.with_scale("10m"),
            facecolor="lightgrey",
            edgecolor="face",
            zorder=3,
        )

    if coastline:
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), zorder=4)

    if borders:
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), zorder=4)

    if states:
        ax.add_feature(cfeature.STATES.with_scale("10m"), zorder=4)

    kwargs = {
        "linewidth": 1.0,
        "color": "black",
        "alpha": 0.5,
        "linestyle": "--",
        "zorder": 100,
    }

    if gridline_kwargs is not None:
        kwargs.update(gridline_kwargs)

    gl = ax.gridlines(
        crs=gridline_crs,
        draw_labels=draw_labels,
        **kwargs,
    )

    if gridline_step is not None and extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent

        gl.xlocator = mticker.FixedLocator(
            np.arange(lon_min, lon_max + gridline_step, gridline_step)
        )
        gl.ylocator = mticker.FixedLocator(
            np.arange(lat_min, lat_max + gridline_step, gridline_step)
        )

    if draw_labels:
        gl.top_labels = False
        gl.right_labels = False

        # Important for projected maps such as AlbersEqualArea.
        # Prevents Cartopy from trying to place labels on the curved
        # geographic boundary, which can trigger Shapely LinearRing errors.
        gl.geo_labels = False
        
        # Space between labels and map boundary, in points.
        gl.xpadding = 15
        gl.ypadding = 5

        gl.xformatter = cticker.LongitudeFormatter()
        gl.yformatter = cticker.LatitudeFormatter()

        gl.xlabel_style = {
            "size": 9,
            "rotation": 0,
        }
        gl.ylabel_style = {
            "size": 9,
            "rotation": 0,
        }

    return gl


def plot_violin_by_year(
    yearly_data: np.ndarray | Sequence[Sequence[float]],
    *,
    years: Sequence[int | str] | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    output_path: PathLike | None = None,
    show: bool = True,
    dpi: int = 300,
    figsize: tuple[float, float] = (14, 7),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot yearly violin distributions.
    """
    data = [np.asarray(values, dtype=float) for values in yearly_data]
    data = [values[np.isfinite(values)] for values in data]

    if years is None:
        years = list(range(1, len(data) + 1))

    if len(years) != len(data):
        raise ValueError(
            f"years and yearly_data must have the same length. "
            f"Got {len(years)} years and {len(data)} data groups."
        )

    fig, ax = plt.subplots(figsize=figsize)

    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=False,
        showextrema=True,
        bw_method="scott",
    )

    for body in parts["bodies"]:
        body.set_alpha(0.8)

    quartile1 = np.array(
        [np.percentile(values, 25) if values.size else np.nan for values in data]
    )
    medians = np.array(
        [np.percentile(values, 50) if values.size else np.nan for values in data]
    )
    quartile3 = np.array(
        [np.percentile(values, 75) if values.size else np.nan for values in data]
    )

    positions = np.arange(1, len(medians) + 1)

    ax.scatter(positions, medians, marker="o", s=25, zorder=3)
    ax.vlines(positions, quartile1, quartile3, linestyle="-", linewidth=2)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(year) for year in years], rotation=45)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("Year")
    ax.set_xlim(0.25, len(data) + 0.75)

    fig.tight_layout()

    _save_or_show(fig, output_path=output_path, show=show, dpi=dpi)

    return fig, ax


def plot_2d_map(
    da: xr.DataArray,
    *,
    lon: xr.DataArray | np.ndarray | None = None,
    lat: xr.DataArray | np.ndarray | None = None,
    extent: Sequence[float] | None = None,
    levels: Sequence[float] | int | None = None,
    cmap: str | mpl.colors.Colormap | None = None,
    title: str | None = None,
    colorbar_label: str | None = None,
    output_path: PathLike | None = None,
    show: bool = True,
    dpi: int = 300,
    figsize: tuple[float, float] = (8, 8),
    projection: ccrs.Projection = ccrs.PlateCarree(),
    data_crs: ccrs.Projection = ccrs.PlateCarree(),
    centered: bool = False,
    gridline_step: int | float | None = None,
    map_features: dict[str, Any] | None = None,
    contour: bool = False,
    contour_levels: Sequence[float] | None = None,
    contour_kwargs: dict[str, Any] | None = None,
    quiver_u: xr.DataArray | np.ndarray | None = None,
    quiver_v: xr.DataArray | np.ndarray | None = None,
    quiver_lon: xr.DataArray | np.ndarray | None = None,
    quiver_lat: xr.DataArray | np.ndarray | None = None,
    quiver_stride: int | None = None,
    quiver_max_vectors: int = 25,
    quiver_scale: float | None = None,
    quiver_width: float | None = None,
    quiver_kwargs: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one 2D map from an xarray DataArray.
    """
    x, y = _prepare_lon_lat_mesh(da, lon=lon, lat=lat)
    z = _dataarray_to_plot_values(da)

    draw_labels, extent_crs, gridline_crs, feature_kwargs = _split_map_feature_kwargs(
        map_features,
        default_draw_labels=False,
        default_extent_crs=data_crs,
        default_gridline_crs=ccrs.PlateCarree(),
    )

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)

    norm = mpl.colors.CenteredNorm() if centered else None

    cs = ax.contourf(
        x,
        y,
        z,
        levels=levels,
        cmap=cmap,
        norm=norm,
        transform=data_crs,
        extend="both",
    )

    if contour:
        line_levels = contour_levels if contour_levels is not None else levels
        kwargs = dict(contour_kwargs or {})

        if line_levels is not None:
            contour_lines = ax.contour(
                x,
                y,
                z,
                levels=line_levels,
                transform=data_crs,
                **kwargs,
            )
            ax.clabel(contour_lines, inline=True, fontsize=8)

    if quiver_u is not None and quiver_v is not None:
        qlon = quiver_lon if quiver_lon is not None else x
        qlat = quiver_lat if quiver_lat is not None else y

        plot_quiver(
            ax,
            quiver_u,
            quiver_v,
            lon=qlon,
            lat=qlat,
            stride=quiver_stride,
            max_vectors=quiver_max_vectors,
            scale=quiver_scale,
            width=quiver_width,
            data_crs=data_crs,
            quiver_kwargs=quiver_kwargs,
        )

    add_map_features(
        ax,
        extent=extent,
        extent_crs=extent_crs,
        gridline_crs=gridline_crs,
        gridline_step=gridline_step,
        draw_labels=draw_labels,
        **feature_kwargs,
    )

    if title is not None:
        ax.set_title(title)

    colorbar = fig.colorbar(
        cs,
        ax=ax,
        orientation="horizontal",
        pad=0.06,
        shrink=0.75,
    )

    if colorbar_label is not None:
        colorbar.set_label(colorbar_label)

    fig.tight_layout()

    _save_or_show(fig, output_path=output_path, show=show, dpi=dpi)

    return fig, ax


def _get_group_dimension(
    data: xr.DataArray,
    *,
    group_dim: str | None = None,
) -> str:
    """
    Infer the subplot/group dimension.
    """
    if group_dim is not None:
        if group_dim not in data.dims and group_dim not in data.coords:
            raise KeyError(f"group_dim={group_dim!r} was not found.")
        return group_dim

    candidates = ("season", "custom_season", "month", "year")

    for candidate in candidates:
        if candidate in data.dims or candidate in data.coords:
            return candidate

    excluded = {
        "lat",
        "lon",
        "latitude",
        "longitude",
        "nav_lat",
        "nav_lon",
        "x",
        "y",
        "depth",
        "deptht",
        "depthu",
        "depthv",
        "depthw",
    }

    for dim in data.dims:
        if dim not in excluded:
            return dim

    raise ValueError(
        "Could not infer group dimension. "
        "Pass group_dim explicitly, for example group_dim='month'."
    )


def _group_labels(data: xr.DataArray, group_dim: str) -> list[str]:
    """
    Create subplot labels for the group dimension.
    """
    values = data[group_dim].values

    month_names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    if group_dim == "month":
        return [month_names.get(int(value), str(value)) for value in values]

    return [str(value) for value in values]


def _subplot_grid(n_panels: int) -> tuple[int, int]:
    """
    Choose a reasonable subplot grid.
    """
    if n_panels <= 4:
        return 2, 2

    if n_panels <= 6:
        return 2, 3

    if n_panels <= 9:
        return 3, 3

    if n_panels <= 12:
        return 3, 4

    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))

    return nrows, ncols


def _infer_time_span_from_group(group_dim: str, n_panels: int) -> str:
    """
    Infer subplot type from group dimension and number of panels.
    """
    if group_dim == "month" or n_panels == 12:
        return "monthly"

    if group_dim in {"season", "custom_season"} or n_panels == 4:
        return "seasonal"

    if group_dim == "year":
        return "yearly"

    return "generic"


def _apply_subplot_gridline_labels(
    gl,
    *,
    panel_index: int,
    nrows: int,
    ncols: int,
    time_span: str,
) -> None:
    """
    Hide repeated gridline labels in subplot maps.
    """
    row_index = panel_index // ncols
    col_index = panel_index % ncols

    if time_span in {"monthly", "seasonal", "yearly", "generic"}:
        if row_index != nrows - 1:
            gl.bottom_labels = False

        if col_index != 0:
            gl.left_labels = False


def _subplot_default_figsize(
    time_span: str,
    *,
    nrows: int,
    ncols: int,
    n_panels: int,
) -> tuple[float, float]:
    """
    Default figure sizes tuned for map subplots.

    These are intentionally smaller/tighter than the first draft.
    """
    if time_span == "seasonal":
        return 8.6, 7.2

    if time_span == "monthly":
        return 13.5, 8.8

    if time_span == "yearly":
        # If there are many years, keep each panel compact.
        return max(10.0, 3.1 * ncols), max(7.0, 2.7 * nrows)

    return max(8.0, 3.3 * ncols), max(6.5, 3.0 * nrows)


def _subplot_adjustment_kwargs(time_span: str) -> dict[str, float]:
    """
    Return subplot spacing for map panels.

    Coordinates are figure fractions:
    left/right/bottom/top define the panel area.
    The colorbar is added separately with fig.add_axes().
    """
    if time_span == "seasonal":
        return {
            "top": 0.90,
            "bottom": 0.08,
            "left": 0.07,
            "right": 0.88,
            "hspace": 0.12,
            "wspace": 0.04,
        }

    if time_span == "monthly":
        return {
            "top": 0.88,
            "bottom": 0.11,
            "left": 0.06,
            "right": 0.9,
            "hspace": 0.1,
            "wspace": 0.1,
        }

    if time_span == "yearly":
        return {
            "top": 0.91,
            "bottom": 0.07,
            "left": 0.055,
            "right": 0.89,
            "hspace": 0.035,
            "wspace": 0.035,
        }

    return {
        "top": 0.90,
        "bottom": 0.08,
        "left": 0.065,
        "right": 0.89,
        "hspace": 0.08,
        "wspace": 0.05,
    }


def _subplot_colorbar_position(time_span: str) -> tuple[float, float, float, float]:
    """
    Default vertical colorbar position.

    Format: [left, bottom, width, height]
    """
    if time_span == "seasonal":
        return 0.91, 0.16, 0.020, 0.62

    if time_span == "monthly":
        return 0.915, 0.14, 0.018, 0.68

    if time_span == "yearly":
        return 0.915, 0.14, 0.018, 0.68

    return 0.915, 0.15, 0.018, 0.66


def plot_2d_subplots(
    da: xr.DataArray,
    *,
    group_dim: str | None = None,
    lon: xr.DataArray | np.ndarray | None = None,
    lat: xr.DataArray | np.ndarray | None = None,
    extent: Sequence[float] | None = None,
    levels: Sequence[float] | int | None = None,
    cmap: str | mpl.colors.Colormap | None = None,
    colorbar_label: str | None = None,
    title: str | None = None,
    output_path: PathLike | None = None,
    show: bool = True,
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
    layout_kwargs: dict[str, float] | None = None,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    data_crs: ccrs.Projection = ccrs.PlateCarree(),
    centered: bool = False,
    gridline_step: int | float | None = None,
    map_features: dict[str, Any] | None = None,
    colorbar_position: Sequence[float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot seasonal, monthly, annual, or other grouped 2D fields as subplots.
    """
    group = _get_group_dimension(da, group_dim=group_dim)
    labels = _group_labels(da, group)

    n_panels = da.sizes[group]
    nrows, ncols = _subplot_grid(n_panels)
    time_span = _infer_time_span_from_group(group, n_panels)

    if figsize is None:
        figsize = _subplot_default_figsize(
        time_span,
        nrows=nrows,
        ncols=ncols,
        n_panels=n_panels,
    )

    x, y = _prepare_lon_lat_mesh(
        da.isel({group: 0}),
        lon=lon,
        lat=lat,
    )

    draw_labels, extent_crs, gridline_crs, feature_kwargs = _split_map_feature_kwargs(
        map_features,
        default_draw_labels=False,
        default_extent_crs=data_crs,
        default_gridline_crs=ccrs.PlateCarree(),
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": projection},
        constrained_layout=False,
    )

    axes_array = np.asarray(axes).ravel()
    norm = mpl.colors.CenteredNorm() if centered else None

    cs = None

    for index in range(n_panels):
        ax = axes_array[index]
        field = da.isel({group: index})
        z = _dataarray_to_plot_values(field)

        cs = ax.contourf(
            x,
            y,
            z,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=data_crs,
            extend="both",
        )

        gl = add_map_features(
            ax,
            extent=extent,
            extent_crs=extent_crs,
            gridline_crs=gridline_crs,
            gridline_step=gridline_step,
            draw_labels=draw_labels,
            **feature_kwargs,
        )

        ax.set_title(labels[index], fontsize=11, pad=2)

        if draw_labels:
            _apply_subplot_gridline_labels(
                gl,
                panel_index=index,
                nrows=nrows,
                ncols=ncols,
                time_span=time_span,
            )

    for empty_ax in axes_array[n_panels:]:
        empty_ax.remove()

    subplot_adjustments = _subplot_adjustment_kwargs(time_span)

    if layout_kwargs is not None:
        subplot_adjustments.update(layout_kwargs)

    fig.subplots_adjust(**subplot_adjustments)

    if title is not None:
        fig.suptitle(title)

    if cs is not None:
        if colorbar_position is None:
            colorbar_position = _subplot_colorbar_position(time_span)

        cbar_ax = fig.add_axes(colorbar_position)

        colorbar = fig.colorbar(
            cs,
            cax=cbar_ax,
            orientation="vertical",
        )

        if colorbar_label is not None:
            colorbar.set_label(colorbar_label)

    _save_or_show(fig, output_path=output_path, show=show, dpi=dpi)

    return fig, axes


def plot_scalar_with_vectors(
    scalar: xr.DataArray,
    u: xr.DataArray | np.ndarray,
    v: xr.DataArray | np.ndarray,
    *,
    lon: xr.DataArray | np.ndarray | None = None,
    lat: xr.DataArray | np.ndarray | None = None,
    vector_lon: xr.DataArray | np.ndarray | None = None,
    vector_lat: xr.DataArray | np.ndarray | None = None,
    extent: Sequence[float] | None = None,
    levels: Sequence[float] | int | None = None,
    cmap: str | mpl.colors.Colormap | None = None,
    title: str | None = None,
    colorbar_label: str | None = None,
    output_path: PathLike | None = None,
    show: bool = True,
    dpi: int = 300,
    figsize: tuple[float, float] = (8, 8),
    projection: ccrs.Projection = ccrs.PlateCarree(),
    data_crs: ccrs.Projection = ccrs.PlateCarree(),
    centered: bool = False,
    contour: bool = False,
    contour_levels: Sequence[float] | None = None,
    contour_kwargs: dict[str, Any] | None = None,
    quiver_stride: int | None = None,
    quiver_max_vectors: int = 25,
    quiver_scale: float | None = None,
    quiver_width: float | None = None,
    quiver_kwargs: dict[str, Any] | None = None,
    gridline_step: int | float | None = None,
    map_features: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a scalar field with vector arrows on top.
    """
    x, y = _prepare_lon_lat_mesh(scalar, lon=lon, lat=lat)
    z = _dataarray_to_plot_values(scalar)

    draw_labels, extent_crs, gridline_crs, feature_kwargs = _split_map_feature_kwargs(
        map_features,
        default_draw_labels=False,
        default_extent_crs=data_crs,
        default_gridline_crs=ccrs.PlateCarree(),
    )

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)

    norm = mpl.colors.CenteredNorm() if centered else None

    cs = ax.contourf(
        x,
        y,
        z,
        levels=levels,
        cmap=cmap,
        norm=norm,
        transform=data_crs,
        extend="both",
    )

    if contour:
        line_levels = contour_levels if contour_levels is not None else levels
        kwargs = dict(contour_kwargs or {})

        if line_levels is not None:
            contour_lines = ax.contour(
                x,
                y,
                z,
                levels=line_levels,
                transform=data_crs,
                **kwargs,
            )
            ax.clabel(contour_lines, inline=True, fontsize=8)

    qlon = vector_lon if vector_lon is not None else x
    qlat = vector_lat if vector_lat is not None else y

    plot_quiver(
        ax,
        u,
        v,
        lon=qlon,
        lat=qlat,
        stride=quiver_stride,
        max_vectors=quiver_max_vectors,
        scale=quiver_scale,
        width=quiver_width,
        data_crs=data_crs,
        quiver_kwargs=quiver_kwargs,
    )

    add_map_features(
        ax,
        extent=extent,
        extent_crs=extent_crs,
        gridline_crs=gridline_crs,
        gridline_step=gridline_step,
        draw_labels=draw_labels,
        **feature_kwargs,
    )

    if title is not None:
        ax.set_title(title)

    colorbar = fig.colorbar(
        cs,
        ax=ax,
        orientation="horizontal",
        pad=0.06,
        shrink=0.75,
    )

    if colorbar_label is not None:
        colorbar.set_label(colorbar_label)

    fig.tight_layout()

    _save_or_show(fig, output_path=output_path, show=show, dpi=dpi)

    return fig, ax


def plot_vector_map(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    lon: xr.DataArray | np.ndarray | None = None,
    lat: xr.DataArray | np.ndarray | None = None,
    extent: Sequence[float] | None = None,
    levels: Sequence[float] | int | None = None,
    cmap: str | mpl.colors.Colormap | None = None,
    title: str | None = None,
    colorbar_label: str | None = "Speed",
    output_path: PathLike | None = None,
    show: bool = True,
    dpi: int = 300,
    figsize: tuple[float, float] = (8, 8),
    projection: ccrs.Projection = ccrs.PlateCarree(),
    data_crs: ccrs.Projection = ccrs.PlateCarree(),
    quiver_stride: int | None = None,
    quiver_max_vectors: int = 25,
    quiver_scale: float | None = None,
    quiver_width: float | None = None,
    quiver_kwargs: dict[str, Any] | None = None,
    gridline_step: int | float | None = None,
    map_features: dict[str, Any] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot vector speed as background and u/v as arrows.
    """
    speed = np.sqrt(u**2 + v**2)
    speed = speed.rename("speed")

    return plot_scalar_with_vectors(
        speed,
        u,
        v,
        lon=lon,
        lat=lat,
        extent=extent,
        levels=levels,
        cmap=cmap,
        title=title,
        colorbar_label=colorbar_label,
        output_path=output_path,
        show=show,
        dpi=dpi,
        figsize=figsize,
        projection=projection,
        data_crs=data_crs,
        quiver_stride=quiver_stride,
        quiver_max_vectors=quiver_max_vectors,
        quiver_scale=quiver_scale,
        quiver_width=quiver_width,
        quiver_kwargs=quiver_kwargs,
        gridline_step=gridline_step,
        map_features=map_features,
    )


def _slice_group_if_needed(
    data: xr.DataArray | np.ndarray,
    *,
    group_dim: str,
    index: int,
):
    """
    Slice a grouped DataArray if it has the group dimension.
    """
    if isinstance(data, xr.DataArray) and group_dim in data.dims:
        return data.isel({group_dim: index})

    return data


def plot_scalar_vector_subplots(
    scalar: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    group_dim: str | None = None,
    lon: xr.DataArray | np.ndarray | None = None,
    lat: xr.DataArray | np.ndarray | None = None,
    vector_lon: xr.DataArray | np.ndarray | None = None,
    vector_lat: xr.DataArray | np.ndarray | None = None,
    extent: Sequence[float] | None = None,
    levels: Sequence[float] | int | None = None,
    cmap: str | mpl.colors.Colormap | None = None,
    colorbar_label: str | None = None,
    title: str | None = None,
    output_path: PathLike | None = None,
    show: bool = True,
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
    layout_kwargs: dict[str, float] | None = None,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    data_crs: ccrs.Projection = ccrs.PlateCarree(),
    centered: bool = False,
    contour: bool = False,
    contour_levels: Sequence[float] | None = None,
    contour_kwargs: dict[str, Any] | None = None,
    quiver_stride: int | None = None,
    quiver_max_vectors: int = 25,
    quiver_scale: float | None = None,
    quiver_width: float | None = None,
    quiver_kwargs: dict[str, Any] | None = None,
    gridline_step: int | float | None = None,
    map_features: dict[str, Any] | None = None,
    colorbar_position: Sequence[float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot grouped scalar fields with grouped vector arrows.
    """
    group = _get_group_dimension(scalar, group_dim=group_dim)
    labels = _group_labels(scalar, group)

    n_panels = scalar.sizes[group]
    nrows, ncols = _subplot_grid(n_panels)
    time_span = _infer_time_span_from_group(group, n_panels)

    if figsize is None:
        figsize = _subplot_default_figsize(
        time_span,
        nrows=nrows,
        ncols=ncols,
        n_panels=n_panels,
    )

    x, y = _prepare_lon_lat_mesh(
        scalar.isel({group: 0}),
        lon=lon,
        lat=lat,
    )

    qlon = vector_lon if vector_lon is not None else x
    qlat = vector_lat if vector_lat is not None else y

    draw_labels, extent_crs, gridline_crs, feature_kwargs = _split_map_feature_kwargs(
        map_features,
        default_draw_labels=False,
        default_extent_crs=data_crs,
        default_gridline_crs=ccrs.PlateCarree(),
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": projection},
        constrained_layout=False,
    )

    axes_array = np.asarray(axes).ravel()
    norm = mpl.colors.CenteredNorm() if centered else None
    contour_options = dict(contour_kwargs or {})

    cs = None

    for index in range(n_panels):
        ax = axes_array[index]

        scalar_plot = scalar.isel({group: index})
        z = _dataarray_to_plot_values(scalar_plot)

        cs = ax.contourf(
            x,
            y,
            z,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=data_crs,
            extend="both",
        )

        if contour:
            line_levels = contour_levels if contour_levels is not None else levels

            if line_levels is not None:
                contour_lines = ax.contour(
                    x,
                    y,
                    z,
                    levels=line_levels,
                    transform=data_crs,
                    **contour_options,
                )
                ax.clabel(contour_lines, inline=True, fontsize=8)

        u_plot = _slice_group_if_needed(u, group_dim=group, index=index)
        v_plot = _slice_group_if_needed(v, group_dim=group, index=index)
        qlon_plot = _slice_group_if_needed(qlon, group_dim=group, index=index)
        qlat_plot = _slice_group_if_needed(qlat, group_dim=group, index=index)

        plot_quiver(
            ax,
            u_plot,
            v_plot,
            lon=qlon_plot,
            lat=qlat_plot,
            stride=quiver_stride,
            max_vectors=quiver_max_vectors,
            scale=quiver_scale,
            width=quiver_width,
            data_crs=data_crs,
            quiver_kwargs=quiver_kwargs,
        )

        gl = add_map_features(
            ax,
            extent=extent,
            extent_crs=extent_crs,
            gridline_crs=gridline_crs,
            gridline_step=gridline_step,
            draw_labels=draw_labels,
            **feature_kwargs,
        )

        ax.set_title(labels[index], fontsize=11, pad=2)

        if draw_labels:
            _apply_subplot_gridline_labels(
                gl,
                panel_index=index,
                nrows=nrows,
                ncols=ncols,
                time_span=time_span,
            )

    for empty_ax in axes_array[n_panels:]:
        empty_ax.remove()

    subplot_adjustments = _subplot_adjustment_kwargs(time_span)

    if layout_kwargs is not None:
        subplot_adjustments.update(layout_kwargs)

    fig.subplots_adjust(**subplot_adjustments)

    if title is not None:
        fig.suptitle(title)

    if cs is not None:
        if colorbar_position is None:
            colorbar_position = _subplot_colorbar_position(time_span)

        cbar_ax = fig.add_axes(colorbar_position)

        colorbar = fig.colorbar(
            cs,
            cax=cbar_ax,
            orientation="vertical",
        )

        if colorbar_label is not None:
            colorbar.set_label(colorbar_label)

    _save_or_show(fig, output_path=output_path, show=show, dpi=dpi)

    return fig, axes
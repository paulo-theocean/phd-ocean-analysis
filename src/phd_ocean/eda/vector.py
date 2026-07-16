from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr


DirectionConvention = Literal["math", "bearing", "coming_from"]
DirectionUnits = Literal["degrees", "radians"]


def compute_speed(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    name: str = "speed",
) -> xr.DataArray:
    """
    Compute vector speed from zonal and meridional components.

    Parameters
    ----------
    u
        Zonal/eastward component.
    v
        Meridional/northward component.
    name
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        Vector speed: sqrt(u**2 + v**2).
    """
    speed = np.sqrt(u**2 + v**2)
    speed = speed.rename(name)

    speed.attrs.update(
        {
            "long_name": "vector speed",
            "description": "sqrt(u**2 + v**2)",
        }
    )

    return speed


def compute_direction(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    convention: DirectionConvention = "bearing",
    units: DirectionUnits = "degrees",
    name: str = "direction",
) -> xr.DataArray:
    """
    Compute vector direction from zonal and meridional components.

    Parameters
    ----------
    u
        Zonal/eastward component.
    v
        Meridional/northward component.
    convention
        Direction convention:

        - 'math':
            0° = East, 90° = North.
            Uses arctan2(v, u).

        - 'bearing':
            0° = North, 90° = East.
            This is often easier for geographic interpretation.
            Uses arctan2(u, v).

        - 'coming_from':
            Meteorological-style direction.
            Direction from which the vector comes.
            This is bearing + 180°.

    units
        'degrees' or 'radians'.
    name
        Name of the output DataArray.

    Returns
    -------
    xr.DataArray
        Direction field.
    """
    if convention == "math":
        angle_rad = np.arctan2(v, u)

    elif convention == "bearing":
        angle_rad = np.arctan2(u, v)

    elif convention == "coming_from":
        angle_rad = np.arctan2(u, v) + np.pi

    else:
        raise ValueError(
            "convention must be 'math', 'bearing', or 'coming_from'."
        )

    if units == "degrees":
        direction = (np.degrees(angle_rad) + 360) % 360
        attrs = {
            "units": "degrees",
            "valid_min": 0.0,
            "valid_max": 360.0,
        }

    elif units == "radians":
        direction = angle_rad % (2 * np.pi)
        attrs = {
            "units": "radians",
            "valid_min": 0.0,
            "valid_max": float(2 * np.pi),
        }

    else:
        raise ValueError("units must be 'degrees' or 'radians'.")

    direction = direction.rename(name)

    direction.attrs.update(
        {
            "long_name": "vector direction",
            "direction_convention": convention,
            **attrs,
        }
    )

    return direction


def compute_vector_diagnostics(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    speed_name: str = "speed",
    direction_name: str = "direction",
    direction_convention: DirectionConvention = "bearing",
    direction_units: DirectionUnits = "degrees",
) -> xr.Dataset:
    """
    Compute speed and direction from vector components.

    Returns
    -------
    xr.Dataset
        Dataset with u, v, speed, and direction.
    """
    speed = compute_speed(
        u,
        v,
        name=speed_name,
    )

    direction = compute_direction(
        u,
        v,
        convention=direction_convention,
        units=direction_units,
        name=direction_name,
    )

    return xr.Dataset(
        {
            "u": u,
            "v": v,
            speed_name: speed,
            direction_name: direction,
        }
    )


def compute_vector_diagnostics_from_dataset(
    ds: xr.Dataset,
    *,
    u_name: str,
    v_name: str,
    speed_name: str = "speed",
    direction_name: str = "direction",
    direction_convention: DirectionConvention = "bearing",
    direction_units: DirectionUnits = "degrees",
) -> xr.Dataset:
    """
    Compute vector diagnostics from a Dataset containing u and v variables.

    Parameters
    ----------
    ds
        Dataset containing vector components.
    u_name
        Name of the zonal/eastward component.
    v_name
        Name of the meridional/northward component.

    Returns
    -------
    xr.Dataset
        Dataset with u, v, speed, and direction.
    """
    if u_name not in ds:
        raise KeyError(f"u variable {u_name!r} was not found in the dataset.")

    if v_name not in ds:
        raise KeyError(f"v variable {v_name!r} was not found in the dataset.")

    return compute_vector_diagnostics(
        ds[u_name],
        ds[v_name],
        speed_name=speed_name,
        direction_name=direction_name,
        direction_convention=direction_convention,
        direction_units=direction_units,
    )


def decompose_speed_direction(
    speed: xr.DataArray,
    direction: xr.DataArray,
    *,
    convention: DirectionConvention = "bearing",
    units: DirectionUnits = "degrees",
    u_name: str = "u",
    v_name: str = "v",
) -> xr.Dataset:
    """
    Convert speed and direction back into u and v components.

    Parameters
    ----------
    speed
        Vector speed.
    direction
        Direction field.
    convention
        Direction convention used by the direction field.
    units
        Direction units.

    Returns
    -------
    xr.Dataset
        Dataset containing u and v.
    """
    if units == "degrees":
        angle = np.deg2rad(direction)
    elif units == "radians":
        angle = direction
    else:
        raise ValueError("units must be 'degrees' or 'radians'.")

    if convention == "math":
        u = speed * np.cos(angle)
        v = speed * np.sin(angle)

    elif convention == "bearing":
        u = speed * np.sin(angle)
        v = speed * np.cos(angle)

    elif convention == "coming_from":
        toward_angle = angle - np.pi
        u = speed * np.sin(toward_angle)
        v = speed * np.cos(toward_angle)

    else:
        raise ValueError(
            "convention must be 'math', 'bearing', or 'coming_from'."
        )

    return xr.Dataset(
        {
            u_name: u.rename(u_name),
            v_name: v.rename(v_name),
        }
    )
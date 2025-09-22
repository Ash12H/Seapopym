"""Defines the structure of the coordinates of the data."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING

import cf_xarray.units  # noqa: F401
import xarray as xr

from seapopym.standard.labels import CoordinatesLabels

if TYPE_CHECKING:
    import numpy as np


def list_available_dims(data: xr.Dataset | xr.DataArray) -> list[str]:
    """Return the standard name of all available coordinates in the data."""
    return [coord for coord in CoordinatesLabels.ordered() if coord in data.cf]


def reorder_dims(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Follow the standard order of dimensions for a xarray.Dataset or xarray.DataArray.

    .. deprecated:: 2024.12
        Use CoordinatesLabels.order_data() instead. This function will be removed in a future version.
    """
    warnings.warn(
        "reorder_dims() is deprecated. Use CoordinatesLabels.order_data() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return CoordinatesLabels.order_data(data)


# Deprecated wrapper functions - use CoordinateAuthority directly instead

def new_latitude(latitude_data: np.ndarray) -> xr.DataArray:
    """Create a new latitude coordinate with standardized Y name.

    .. deprecated:: 2024.12
        Use CoordinateAuthority.get_coordinate_attrs() or the registered factory instead.
        This function will be removed in a future version.
    """
    warnings.warn(
        "new_latitude() is deprecated. Use CoordinateAuthority registry instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from seapopym.standard.coordinate_authority import create_latitude_coordinate
    return create_latitude_coordinate(latitude_data)


def new_longitude(longitude_data: Iterable) -> xr.DataArray:
    """Create a new longitude coordinate with standardized X name.

    .. deprecated:: 2024.12
        Use CoordinateAuthority.get_coordinate_attrs() or the registered factory instead.
        This function will be removed in a future version.
    """
    warnings.warn(
        "new_longitude() is deprecated. Use CoordinateAuthority registry instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from seapopym.standard.coordinate_authority import create_longitude_coordinate
    return create_longitude_coordinate(longitude_data)


def new_layer(layer_data: Iterable | None = None) -> xr.DataArray:
    """Create a new layer coordinate.

    .. deprecated:: 2024.12
        Use CoordinateAuthority.get_coordinate_attrs() or the registered factory instead.
        This function will be removed in a future version.
    """
    warnings.warn(
        "new_layer() is deprecated. Use CoordinateAuthority registry instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from seapopym.standard.coordinate_authority import create_layer_coordinate
    return create_layer_coordinate(layer_data)


def new_time(time_data: Iterable) -> xr.DataArray:
    """Create a new time coordinate with standardized T name.

    .. deprecated:: 2024.12
        Use CoordinateAuthority.get_coordinate_attrs() or the registered factory instead.
        This function will be removed in a future version.
    """
    warnings.warn(
        "new_time() is deprecated. Use CoordinateAuthority registry instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from seapopym.standard.coordinate_authority import create_time_coordinate
    return create_time_coordinate(time_data)


def new_cohort(cohort_data: Iterable) -> xr.DataArray:
    """Create a new cohort coordinate.

    .. deprecated:: 2024.12
        Use CoordinateAuthority.get_coordinate_attrs() or the registered factory instead.
        This function will be removed in a future version.
    """
    warnings.warn(
        "new_cohort() is deprecated. Use CoordinateAuthority registry instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from seapopym.standard.coordinate_authority import create_cohort_coordinate
    return create_cohort_coordinate(cohort_data)

"""
Coordinate Authority for managing coordinate attributes and integrity.

This module implements the CoordinateAuthority pattern to ensure coordinate
attributes (especially axis labels like "T" for time) are preserved during
both sequential and parallel execution paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import xarray as xr
from attrs import frozen

from seapopym.standard.labels import CoordinatesLabels, SeaLayers

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy as np
    import xarray as xr


# Coordinate Factory Functions
# These replace the functions previously in coordinates.py

def create_latitude_coordinate(latitude_data: np.ndarray) -> xr.DataArray:
    """Create a new latitude coordinate with standardized Y name."""
    attributs = {"long_name": "latitude", "standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
    return xr.DataArray(
        coords=[("Y", latitude_data, attributs)],
        dims=["Y"],
    ).coords["Y"]


def create_longitude_coordinate(longitude_data: Iterable) -> xr.DataArray:
    """Create a new longitude coordinate with standardized X name."""
    attributs = {"long_name": "longitude", "standard_name": "longitude", "units": "degrees_east", "axis": "X"}
    return xr.DataArray(
        coords=[("X", longitude_data, attributs)],
        dims=["X"],
    ).coords["X"]


def create_layer_coordinate(layer_data: Iterable | None = None) -> xr.DataArray:
    """Create a new layer coordinate."""
    if layer_data is None:
        layer_data = [layer.depth for layer in SeaLayers]
    attributs = {
        "long_name": "layer",
        "standard_name": "layer",
        "positive": "down",
        "axis": "Z",
        "flag_values": str(layer_data),
        "flag_meanings": " ".join([layer.standard_name for layer in SeaLayers]),
    }
    return xr.DataArray(coords=(("Z", layer_data, attributs),), dims=["Z"]).coords["Z"]


def create_time_coordinate(time_data: Iterable) -> xr.DataArray:
    """Create a new time coordinate with standardized T name."""
    return xr.DataArray(
        coords=[("T", time_data, {"long_name": "time", "standard_name": "time", "axis": "T"})], dims=["T"]
    ).coords["T"]


def create_cohort_coordinate(cohort_data: Iterable) -> xr.DataArray:
    """Create a new cohort coordinate."""
    attributs = {"long_name": "cohort", "standard_name": "cohort"}
    return xr.DataArray(
        coords=[("cohort", cohort_data, attributs)],
        dims=["cohort"],
    ).coords["cohort"]


@frozen
class CoordinateAuthority:
    """
    Central authority for coordinate management and validation.

    Uses a registry pattern to manage coordinate factories, making it extensible
    and maintainable. Ensures coordinate attributes are preserved across xarray
    operations, particularly resolving the asymmetry between sequential and
    parallel execution paths where attributes can be lost.
    """

    _registry: ClassVar[dict[CoordinatesLabels, Callable]] = {}

    @classmethod
    def register_coordinate(cls, label: CoordinatesLabels, factory: Callable) -> None:
        """
        Register a coordinate factory for a given label.

        Args:
            label: Coordinate label from CoordinatesLabels enum
            factory: Function that creates the coordinate with proper attributes

        """
        cls._registry[label] = factory

    @classmethod
    def initialize_defaults(cls) -> None:
        """Initialize the registry with default coordinate factories."""
        cls.register_coordinate(CoordinatesLabels.time, create_time_coordinate)
        cls.register_coordinate(CoordinatesLabels.Y, create_latitude_coordinate)
        cls.register_coordinate(CoordinatesLabels.X, create_longitude_coordinate)
        cls.register_coordinate(CoordinatesLabels.Z, create_layer_coordinate)
        cls.register_coordinate(CoordinatesLabels.cohort, create_cohort_coordinate)

    @classmethod
    def get_registered_coordinates(cls) -> tuple[CoordinatesLabels, ...]:
        """Get all registered coordinate labels."""
        return tuple(cls._registry.keys())

    def validate_coordinates(self, data: xr.Dataset) -> xr.Dataset:
        """
        Validate and restore missing coordinate attributes.

        Args:
            data: xarray Dataset to validate

        Returns:
            Dataset with validated and restored coordinate attributes

        """
        validated_data = data.copy()

        for coord_name, coord_data in validated_data.coords.items():
            if coord_name in self._registry:
                # Get expected attributes from factory
                expected_coord = self._registry[coord_name](coord_data.values)

                # Restore missing attributes
                current_attrs = coord_data.attrs.copy()
                for attr_name, attr_value in expected_coord.attrs.items():
                    if attr_name not in current_attrs:
                        current_attrs[attr_name] = attr_value

                # Update coordinate with complete attributes
                validated_data = validated_data.assign_coords({coord_name: coord_data.assign_attrs(current_attrs)})

        return validated_data

    def ensure_coordinate_integrity(self, data: xr.Dataset) -> xr.Dataset:
        """
        Ensure coordinate integrity after xarray operations.

        This method should be called after operations that might lose
        coordinate attributes to restore them according to CF conventions.

        Args:
            data: xarray Dataset that may have lost coordinate attributes

        Returns:
            Dataset with restored coordinate integrity

        """
        return self.validate_coordinates(data)

    def get_coordinate_attrs(self, coord_label: CoordinatesLabels) -> dict[str, Any]:
        """
        Get expected attributes for a coordinate label.

        Args:
            coord_label: Coordinate label from CoordinatesLabels enum

        Returns:
            Dictionary of expected coordinate attributes

        Raises:
            KeyError: If coordinate label is not supported

        """
        if coord_label not in self._registry:
            msg = f"Unsupported coordinate label: {coord_label}"
            raise KeyError(msg)

        # Create dummy coordinate to extract attributes
        dummy_coord = self._registry[coord_label]([0])
        return dummy_coord.attrs.copy()

    def is_coordinate_valid(self, data: xr.Dataset, coord_label: CoordinatesLabels) -> bool:
        """
        Check if a coordinate has all required attributes.

        Args:
            data: xarray Dataset to check
            coord_label: Coordinate label to validate

        Returns:
            True if coordinate has all required attributes, False otherwise

        """
        if coord_label not in data.coords:
            return False

        try:
            expected_attrs = self.get_coordinate_attrs(coord_label)
            current_attrs = data.coords[coord_label].attrs

            # Check if all expected attributes are present
            return all(attr in current_attrs for attr in expected_attrs)
        except KeyError:
            return False


# Global instance for convenience
coordinate_authority = CoordinateAuthority()

# Initialize default coordinate factories
CoordinateAuthority.initialize_defaults()

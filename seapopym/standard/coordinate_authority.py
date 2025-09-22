"""
Coordinate Authority for managing coordinate attributes and integrity.

This module implements the CoordinateAuthority pattern to ensure coordinate
attributes (especially axis labels like "T" for time) are preserved during
both sequential and parallel execution paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from attrs import frozen

from seapopym.standard.coordinates import new_cohort, new_latitude, new_layer, new_longitude, new_time
from seapopym.standard.labels import CoordinatesLabels

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr


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
        cls.register_coordinate(CoordinatesLabels.time, new_time)
        cls.register_coordinate(CoordinatesLabels.Y, new_latitude)
        cls.register_coordinate(CoordinatesLabels.X, new_longitude)
        cls.register_coordinate(CoordinatesLabels.Z, new_layer)
        cls.register_coordinate(CoordinatesLabels.cohort, new_cohort)

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

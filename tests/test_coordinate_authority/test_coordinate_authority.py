"""Tests for CoordinateAuthority implementation.

Tests validate that coordinate attributes are preserved across operations
and that sequential/parallel execution modes produce identical results.
"""

import numpy as np
import pytest
import xarray as xr

from seapopym.standard.coordinate_authority import CoordinateAuthority, coordinate_authority
from seapopym.standard.coordinates import new_cohort, new_latitude, new_longitude, new_time
from seapopym.standard.labels import CoordinatesLabels


class TestCoordinateAuthority:
    """Test the CoordinateAuthority class functionality."""

    def test_registry_pattern(self):
        """Test the registry pattern functionality."""
        # Save original factory for cleanup
        original_factory = coordinate_authority._registry.get(CoordinatesLabels.time)

        try:
            # Test that we can register new coordinate types
            def custom_time_factory(data):
                return new_time(data).assign_attrs(custom_attr="test")

            CoordinateAuthority.register_coordinate(CoordinatesLabels.time, custom_time_factory)

            # Test that the custom factory is used
            attrs = coordinate_authority.get_coordinate_attrs(CoordinatesLabels.time)
            assert "custom_attr" in attrs
            assert attrs["custom_attr"] == "test"

        finally:
            # Always restore original factory
            if original_factory:
                CoordinateAuthority.register_coordinate(CoordinatesLabels.time, original_factory)

    def test_coordinate_authority_initialization(self):
        """Test that CoordinateAuthority initializes correctly."""
        # Check that all expected coordinate factories are present in the global instance
        expected_coords = {
            CoordinatesLabels.time,
            CoordinatesLabels.Y,
            CoordinatesLabels.X,
            CoordinatesLabels.Z,
            CoordinatesLabels.cohort,
        }
        assert set(coordinate_authority.get_registered_coordinates()) == expected_coords

    def test_get_coordinate_attrs(self):
        """Test getting coordinate attributes."""
        authority = coordinate_authority

        # Test time coordinate attributes
        time_attrs = authority.get_coordinate_attrs(CoordinatesLabels.time)
        assert "axis" in time_attrs
        assert time_attrs["axis"] == "T"
        assert "standard_name" in time_attrs
        assert time_attrs["standard_name"] == "time"

        # Test latitude coordinate attributes
        lat_attrs = authority.get_coordinate_attrs(CoordinatesLabels.Y)
        assert "axis" in lat_attrs
        assert lat_attrs["axis"] == "Y"
        assert "units" in lat_attrs
        assert lat_attrs["units"] == "degrees_north"

    def test_get_coordinate_attrs_invalid_label(self):
        """Test error handling for invalid coordinate labels."""
        authority = coordinate_authority

        with pytest.raises(KeyError, match="Unsupported coordinate label"):
            authority.get_coordinate_attrs("invalid_label")

    def test_validate_coordinates_restores_missing_attrs(self):
        """Test that validate_coordinates restores missing coordinate attributes."""
        authority = coordinate_authority

        # Create a dataset with coordinates missing some attributes
        time_data = np.array([1, 2, 3])
        lat_data = np.array([10.0, 20.0, 30.0])
        lon_data = np.array([100.0, 110.0, 120.0])

        # Create coordinates with missing attributes
        time_coord = xr.DataArray(time_data, dims=["T"], name="T")
        lat_coord = xr.DataArray(lat_data, dims=["Y"], name="Y", attrs={"units": "degrees_north"})  # missing axis
        lon_coord = xr.DataArray(lon_data, dims=["X"], name="X")  # missing all attributes

        # Create dataset
        dataset = xr.Dataset(
            coords={"T": time_coord, "Y": lat_coord, "X": lon_coord}
        )

        # Validate coordinates
        validated = authority.validate_coordinates(dataset)

        # Check that missing attributes were restored
        assert validated.coords["T"].attrs["axis"] == "T"
        assert validated.coords["T"].attrs["standard_name"] == "time"
        assert validated.coords["Y"].attrs["axis"] == "Y"
        assert validated.coords["Y"].attrs["units"] == "degrees_north"  # preserved
        assert validated.coords["X"].attrs["axis"] == "X"
        assert validated.coords["X"].attrs["units"] == "degrees_east"

    def test_ensure_coordinate_integrity(self):
        """Test that ensure_coordinate_integrity works as expected."""
        authority = coordinate_authority

        # Create dataset with proper coordinates
        time_coord = new_time([1, 2, 3])
        lat_coord = new_latitude([10.0, 20.0])

        dataset = xr.Dataset(
            coords={"T": time_coord, "Y": lat_coord}
        )

        # Remove some attributes to simulate loss during operations
        corrupted_dataset = dataset.copy()
        corrupted_dataset.coords["T"].attrs.clear()

        # Ensure integrity
        restored = authority.ensure_coordinate_integrity(corrupted_dataset)

        # Check that attributes were restored
        assert restored.coords["T"].attrs["axis"] == "T"
        assert restored.coords["T"].attrs["standard_name"] == "time"

    def test_is_coordinate_valid(self):
        """Test coordinate validation checking."""
        # Use the global instance that has been initialized
        authority = coordinate_authority

        # Create dataset with valid time coordinate
        time_coord = new_time([1, 2, 3])
        dataset_valid = xr.Dataset(coords={"T": time_coord})

        # Create dataset with invalid time coordinate (missing attributes)
        time_coord_invalid = xr.DataArray([1, 2, 3], dims=["T"], name="T")
        dataset_invalid = xr.Dataset(coords={"T": time_coord_invalid})

        # Create dataset without the coordinate
        dataset_missing = xr.Dataset()

        # Test validation
        assert authority.is_coordinate_valid(dataset_valid, CoordinatesLabels.time) is True
        assert authority.is_coordinate_valid(dataset_invalid, CoordinatesLabels.time) is False
        assert authority.is_coordinate_valid(dataset_missing, CoordinatesLabels.time) is False

    def test_extensibility(self):
        """Test that the registry pattern allows easy extension."""
        # Save original factory for cleanup
        original_factory = coordinate_authority._registry.get(CoordinatesLabels.cohort)

        try:
            # Test adding a completely new coordinate type
            def custom_coord_factory(data):
                return xr.DataArray(
                    data,
                    dims=["custom"],
                    attrs={"custom_attr": "value", "standard_name": "custom_coordinate"}
                )

            # Register new coordinate type (using existing enum value for simplicity)
            CoordinateAuthority.register_coordinate(CoordinatesLabels.cohort, custom_coord_factory)

            # Test that the custom factory is used
            attrs = coordinate_authority.get_coordinate_attrs(CoordinatesLabels.cohort)
            assert attrs["custom_attr"] == "value"
            assert attrs["standard_name"] == "custom_coordinate"

        finally:
            # Always restore original factory, even if test fails
            if original_factory:
                CoordinateAuthority.register_coordinate(CoordinatesLabels.cohort, original_factory)

    def test_global_instance(self):
        """Test that the global coordinate_authority instance works."""
        # Test that the global instance is properly initialized
        assert isinstance(coordinate_authority, CoordinateAuthority)

        # Test that it has the same functionality
        time_attrs = coordinate_authority.get_coordinate_attrs(CoordinatesLabels.time)
        assert "axis" in time_attrs
        assert time_attrs["axis"] == "T"


class TestCoordinateAuthorityIntegration:
    """Integration tests for CoordinateAuthority with real datasets."""

    def test_coordinate_integrity_after_operations(self):
        """Test that coordinate integrity is maintained after xarray operations."""
        # Create a proper dataset with CF-compliant coordinates
        time_coord = new_time([1, 2, 3])
        lat_coord = new_latitude([10.0, 20.0])
        lon_coord = new_longitude([100.0, 110.0, 120.0])

        # Create data variable
        data = np.random.random((3, 2, 3))
        dataset = xr.Dataset(
            {
                "temperature": (["T", "Y", "X"], data)
            },
            coords={"T": time_coord, "Y": lat_coord, "X": lon_coord}
        )

        # Perform operations that might lose attributes
        processed = dataset.sel(T=slice(1, 2)).mean("T")

        # Ensure coordinate integrity
        restored = coordinate_authority.ensure_coordinate_integrity(processed)

        # Check that coordinate attributes are preserved
        assert restored.coords["Y"].attrs["axis"] == "Y"
        assert restored.coords["X"].attrs["axis"] == "X"
        assert restored.coords["Y"].attrs["units"] == "degrees_north"
        assert restored.coords["X"].attrs["units"] == "degrees_east"

    def test_coordinate_validation_preserves_existing_attrs(self):
        """Test that validation preserves existing attributes while adding missing ones."""
        # Create coordinate with some custom attributes
        time_data = np.array([1, 2, 3])
        time_coord = xr.DataArray(
            time_data,
            dims=["T"],
            name="T",
            attrs={
                "units": "days since 2000-01-01",
                "calendar": "gregorian",
                "custom_attr": "custom_value"
            }
        )

        dataset = xr.Dataset(coords={"T": time_coord})

        # Validate coordinates
        validated = coordinate_authority.validate_coordinates(dataset)

        # Check that existing attributes are preserved
        assert validated.coords["T"].attrs["units"] == "days since 2000-01-01"
        assert validated.coords["T"].attrs["calendar"] == "gregorian"
        assert validated.coords["T"].attrs["custom_attr"] == "custom_value"

        # Check that missing standard attributes are added
        assert validated.coords["T"].attrs["axis"] == "T"
        assert validated.coords["T"].attrs["standard_name"] == "time"
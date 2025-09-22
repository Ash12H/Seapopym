"""Tests for StandardUnitsRegistry functionality.

Tests validate that the centralized units management system works correctly
and provides a clean API for unit handling, formatting, and validation.
"""

import pytest

from seapopym.standard.units import StandardUnitsLabels, StandardUnitsRegistry


class TestStandardUnitsRegistry:
    """Test the StandardUnitsRegistry class functionality."""

    def test_format_unit_string(self):
        """Test unit string formatting."""
        # Test standard units (note: pint may format units differently than input strings)
        temp_unit = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature)
        assert "celsius" in temp_unit.lower() or "degree" in temp_unit

        assert StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.time) == "day"
        assert StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.height) == "meter"

        biomass_unit = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.biomass)
        assert "kilogram" in biomass_unit and "meter" in biomass_unit

        production_unit = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.production)
        assert "kilogram" in production_unit and "meter" in production_unit and "day" in production_unit

        assert StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.acidity) == "dimensionless"

    def test_get_unit_attrs_basic(self):
        """Test basic unit attributes generation."""
        attrs = StandardUnitsRegistry.get_unit_attrs(StandardUnitsLabels.temperature)

        assert "units" in attrs
        temp_unit = attrs["units"]
        assert "celsius" in temp_unit.lower() or "degree" in temp_unit
        assert len(attrs) == 1

    def test_get_unit_attrs_with_overrides(self):
        """Test unit attributes with additional overrides."""
        attrs = StandardUnitsRegistry.get_unit_attrs(
            StandardUnitsLabels.temperature,
            long_name="Temperature",
            standard_name="air_temperature"
        )

        temp_unit = attrs["units"]
        assert "celsius" in temp_unit.lower() or "degree" in temp_unit
        assert attrs["long_name"] == "Temperature"
        assert attrs["standard_name"] == "air_temperature"
        assert len(attrs) == 3

    def test_get_unit_attrs_override_units(self):
        """Test that overrides can replace the units key."""
        attrs = StandardUnitsRegistry.get_unit_attrs(
            StandardUnitsLabels.temperature,
            units="kelvin"  # Override the default celsius
        )

        assert attrs["units"] == "kelvin"

    def test_get_supported_units(self):
        """Test getting all supported unit labels."""
        supported = StandardUnitsRegistry.get_supported_units()

        # Should return tuple of all StandardUnitsLabels
        assert isinstance(supported, tuple)
        assert len(supported) == len(StandardUnitsLabels)

        # Check that all enum values are present
        expected_units = {
            StandardUnitsLabels.height,
            StandardUnitsLabels.weight,
            StandardUnitsLabels.temperature,
            StandardUnitsLabels.time,
            StandardUnitsLabels.biomass,
            StandardUnitsLabels.production,
            StandardUnitsLabels.acidity,
        }
        assert set(supported) == expected_units

    def test_validate_unit_compatibility_no_units_attr(self):
        """Test validation with object without units attribute."""
        class NoUnitsObject:
            pass

        obj = NoUnitsObject()
        result = StandardUnitsRegistry.validate_unit_compatibility(obj, StandardUnitsLabels.temperature)
        assert result is False

    def test_validate_unit_compatibility_none_units(self):
        """Test validation with None units."""
        class NoneUnitsObject:
            units = None

        obj = NoneUnitsObject()
        result = StandardUnitsRegistry.validate_unit_compatibility(obj, StandardUnitsLabels.temperature)
        assert result is False

    def test_validate_unit_compatibility_matching_units(self):
        """Test validation with matching units."""
        class MatchingUnitsObject:
            def __init__(self, units_str):
                self.units = units_str

        # Use the actual formatted unit string from the system
        expected_unit_str = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature)
        obj = MatchingUnitsObject(expected_unit_str)
        result = StandardUnitsRegistry.validate_unit_compatibility(obj, StandardUnitsLabels.temperature)
        assert result is True

    def test_validate_unit_compatibility_mismatched_units(self):
        """Test validation with mismatched units."""
        class MismatchedUnitsObject:
            def __init__(self, units_str):
                self.units = units_str

        obj = MismatchedUnitsObject("kelvin")
        result = StandardUnitsRegistry.validate_unit_compatibility(obj, StandardUnitsLabels.temperature)
        assert result is False

    def test_validate_unit_compatibility_exception_handling(self):
        """Test validation handles exceptions gracefully."""
        class ExceptionObject:
            @property
            def units(self):
                raise ValueError("Test exception")

        obj = ExceptionObject()
        result = StandardUnitsRegistry.validate_unit_compatibility(obj, StandardUnitsLabels.temperature)
        assert result is False


class TestUnitsIntegration:
    """Integration tests for units system."""

    def test_backward_compatibility_with_old_api(self):
        """Test that new API produces same results as old API."""
        # Old API: str(StandardUnitsLabels.temperature.units)
        old_result = str(StandardUnitsLabels.temperature.units)

        # New API: StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature)
        new_result = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature)

        assert old_result == new_result

    def test_all_standard_units_can_be_formatted(self):
        """Test that all standard units can be formatted without errors."""
        for unit_label in StandardUnitsLabels:
            # Should not raise any exceptions
            result = StandardUnitsRegistry.format_unit_string(unit_label)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_complex_units_formatting(self):
        """Test formatting of complex compound units."""
        # Test biomass: kilogram / meter**2
        biomass_units = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.biomass)
        assert "kilogram" in biomass_units
        assert "meter" in biomass_units

        # Test production: kilogram / meter**2 / day
        production_units = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.production)
        assert "kilogram" in production_units
        assert "meter" in production_units
        assert "day" in production_units

    def test_registry_is_stateless(self):
        """Test that registry methods are stateless and don't interfere."""
        # Multiple calls should return identical results
        result1 = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature)
        result2 = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature)
        assert result1 == result2

        # Different units should be independent
        temp_result = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature)
        time_result = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.time)
        assert temp_result != time_result

    def test_registry_methods_are_static(self):
        """Test that all registry methods can be called without instantiation."""
        # All methods should be callable as static methods (no need to instantiate)
        # Test that methods can be called directly on the class
        result1 = StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature)
        result2 = StandardUnitsRegistry.get_unit_attrs(StandardUnitsLabels.temperature)
        result3 = StandardUnitsRegistry.validate_unit_compatibility(None, StandardUnitsLabels.temperature)
        result4 = StandardUnitsRegistry.get_supported_units()

        # Basic sanity checks
        assert isinstance(result1, str)
        assert isinstance(result2, dict)
        assert isinstance(result3, bool)
        assert isinstance(result4, tuple)
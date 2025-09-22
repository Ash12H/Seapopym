"""A module for handling units in the forcing data following the CF conventions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

# ------------------------------------------------------------------------------------------------------- #
# NOTE(Jules): cf_xarray.units import is necessary to work with cf_xarray complient Datasets. DO NOT REMOVE
import cf_xarray.units  # noqa: F401

# ------------------------------------------------------------------------------------------------------- #
import pint
import pint_xarray  # noqa: F401


class StandardUnitsLabels(StrEnum):
    """Unit of measurement as used in the model."""

    height = "meter"
    weight = "kilogram"
    temperature = "celsius"
    time = "day"
    biomass = "kilogram / meter**2"
    production = "kilogram / meter**2 / day"
    acidity = "dimensionless"

    def __init__(self: StandardUnitsLabels, unit_as_str: str) -> None:
        """Prevent the instantiation of this class."""
        self._units = pint.application_registry(unit_as_str).units

    @property
    def units(self: StandardUnitsLabels) -> pint.Unit:
        """Convert the string unit to the equivalent pint unit."""
        return self._units


class StandardUnitsRegistry:
    """
    Centralized registry for units management and conversion.

    Provides a clean API for unit handling, formatting, and validation
    to replace scattered unit conversions throughout the codebase.

    This registry acts as the single source of truth for all unit operations
    in Seapopym, ensuring consistent unit handling across templates, kernels,
    and data processing functions.

    Key Features:
        - Standardized unit string formatting for xarray attributes
        - Convenient attribute dictionary generation for DataArrays
        - Unit compatibility validation for data objects
        - Complete enumeration of supported units

    Usage Patterns:
        Basic unit formatting:
        >>> StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature)
        'degree_Celsius'

        Creating xarray attributes:
        >>> attrs = StandardUnitsRegistry.get_unit_attrs(
        ...     StandardUnitsLabels.biomass,
        ...     long_name="Biomass density",
        ...     standard_name="biomass_density"
        ... )
        >>> # Use with xarray: data_array.attrs.update(attrs)

        Template integration:
        >>> # In template factories
        >>> attrs = StandardUnitsRegistry.get_unit_attrs(
        ...     StandardUnitsLabels.production,
        ...     standard_name="primary_production_rate"
        ... )

    Thread Safety:
        All methods are static and thread-safe. No instance state is maintained.
    """

    @staticmethod
    def format_unit_string(unit_label: StandardUnitsLabels) -> str:
        """
        Format a unit label to standardized string representation.

        Converts a StandardUnitsLabels enum value to its string representation
        as formatted by the pint units library. This ensures consistent unit
        formatting across all xarray attributes and CF-compliant datasets.

        Args:
            unit_label: Unit label from StandardUnitsLabels enum

        Returns:
            Formatted unit string suitable for xarray attributes.
            Note: Pint may format units differently than the input strings
            (e.g., 'celsius' becomes 'degree_Celsius')

        Raises:
            AttributeError: If unit_label is not a valid StandardUnitsLabels enum

        Examples:
            Basic unit formatting:
            >>> StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.temperature)
            'degree_Celsius'
            >>> StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.time)
            'day'

            Complex compound units:
            >>> StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.biomass)
            'kilogram / meter ** 2'
            >>> StandardUnitsRegistry.format_unit_string(StandardUnitsLabels.production)
            'kilogram / meter ** 2 / day'

        Note:
            This method is the recommended way to get unit strings for xarray
            attributes, replacing direct access to StandardUnitsLabels.units
        """
        return str(unit_label.units)

    @staticmethod
    def get_unit_attrs(unit_label: StandardUnitsLabels, **overrides: Any) -> dict[str, str]:
        """
        Get standardized unit attributes for xarray DataArrays.

        Creates a dictionary of attributes suitable for xarray DataArray.attrs,
        starting with the properly formatted units and allowing additional
        CF-compliant attributes to be specified as overrides.

        Args:
            unit_label: Unit label from StandardUnitsLabels enum
            **overrides: Additional attributes to include or override.
                        Common CF attributes include 'long_name', 'standard_name',
                        'description', etc. The 'units' key can be overridden
                        if a different unit string is needed.

        Returns:
            Dictionary with 'units' key and any additional overrides.
            All values are strings suitable for xarray attributes.

        Examples:
            Basic usage (units only):
            >>> attrs = StandardUnitsRegistry.get_unit_attrs(StandardUnitsLabels.temperature)
            >>> attrs
            {'units': 'degree_Celsius'}

            With CF-compliant metadata:
            >>> attrs = StandardUnitsRegistry.get_unit_attrs(
            ...     StandardUnitsLabels.biomass,
            ...     long_name="Biomass density",
            ...     standard_name="biomass_density",
            ...     description="Dry weight biomass per unit area"
            ... )
            >>> attrs
            {'units': 'kilogram / meter ** 2', 'long_name': 'Biomass density', ...}

            Overriding units:
            >>> attrs = StandardUnitsRegistry.get_unit_attrs(
            ...     StandardUnitsLabels.temperature,
            ...     units="kelvin",  # Override default celsius
            ...     standard_name="sea_water_temperature"
            ... )
            >>> attrs
            {'units': 'kelvin', 'standard_name': 'sea_water_temperature'}

            Template factory usage:
            >>> # Common pattern in template factories
            >>> attrs = StandardUnitsRegistry.get_unit_attrs(
            ...     StandardUnitsLabels.production,
            ...     long_name="Primary production rate",
            ...     standard_name="net_primary_production_of_biomass_expressed_as_carbon"
            ... )

        Note:
            This is the recommended way to create xarray attributes with units,
            ensuring consistency across all DataArrays in the system.
        """
        attrs = {"units": StandardUnitsRegistry.format_unit_string(unit_label)}
        attrs.update(overrides)
        return attrs

    @staticmethod
    def validate_unit_compatibility(value: Any, expected_unit: StandardUnitsLabels) -> bool:
        """
        Validate that a value has compatible units.

        Performs unit compatibility validation by comparing the units attribute
        of a value object against an expected StandardUnitsLabels enum value.
        This validation ensures data consistency across the pipeline.

        Args:
            value: Object to validate. Must have a 'units' attribute (typically
                  xarray DataArray, Dataset, or similar objects with units metadata)
            expected_unit: Expected unit label from StandardUnitsLabels enum

        Returns:
            True if units match exactly, False otherwise.
            Returns False for any objects without units or on validation errors.

        Validation Logic:
            - Checks if value has 'units' attribute
            - Compares string representation of units for exact match
            - Handles exceptions gracefully by returning False

        Examples:
            Validating xarray DataArray:
            >>> import xarray as xr
            >>> data = xr.DataArray([1, 2, 3], attrs={'units': 'degree_Celsius'})
            >>> StandardUnitsRegistry.validate_unit_compatibility(
            ...     data, StandardUnitsLabels.temperature
            ... )
            True

            Invalid units:
            >>> data = xr.DataArray([1, 2, 3], attrs={'units': 'kelvin'})
            >>> StandardUnitsRegistry.validate_unit_compatibility(
            ...     data, StandardUnitsLabels.temperature
            ... )
            False

            Missing units attribute:
            >>> data = xr.DataArray([1, 2, 3])  # No units
            >>> StandardUnitsRegistry.validate_unit_compatibility(
            ...     data, StandardUnitsLabels.temperature
            ... )
            False

            Custom objects:
            >>> class TemperatureData:
            ...     def __init__(self, units_str):
            ...         self.units = units_str
            >>> data = TemperatureData('degree_Celsius')
            >>> StandardUnitsRegistry.validate_unit_compatibility(
            ...     data, StandardUnitsLabels.temperature
            ... )
            True

        Error Handling:
            All exceptions (AttributeError, TypeError, ValueError) are caught
            and result in False return value. This ensures robust validation
            even with malformed or unexpected input objects.

        Future Enhancements:
            This basic string comparison can be extended to support:
            - Pint unit compatibility checking (e.g., celsius vs kelvin)
            - Unit conversion compatibility
            - Dimensional analysis validation

        Note:
            Currently performs exact string matching. Compatible but differently
            formatted units (e.g., 'deg_C' vs 'degree_Celsius') will return False.
        """
        try:
            if not hasattr(value, "units"):
                return False

            expected_pint_unit = expected_unit.units
            actual_units = getattr(value, "units", None)

            if actual_units is None:
                return False

            # Basic string comparison - can be enhanced with pint compatibility checking
            return str(actual_units) == str(expected_pint_unit)
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def get_supported_units() -> tuple[StandardUnitsLabels, ...]:
        """
        Get all supported unit labels.

        Returns a tuple containing all available StandardUnitsLabels enum values,
        providing a complete inventory of units supported by the system.

        Returns:
            Tuple of all available StandardUnitsLabels enum values.
            The tuple is ordered according to the enum definition.

        Examples:
            Getting all supported units:
            >>> units = StandardUnitsRegistry.get_supported_units()
            >>> len(units)
            7
            >>> StandardUnitsLabels.temperature in units
            True

            Iterating over supported units:
            >>> for unit_label in StandardUnitsRegistry.get_supported_units():
            ...     unit_str = StandardUnitsRegistry.format_unit_string(unit_label)
            ...     print(f"{unit_label.name}: {unit_str}")
            height: meter
            weight: kilogram
            temperature: degree_Celsius
            ...

            Validation and enumeration:
            >>> supported = StandardUnitsRegistry.get_supported_units()
            >>> user_unit = "temperature"  # From user input
            >>> if any(unit.name == user_unit for unit in supported):
            ...     unit_label = StandardUnitsLabels[user_unit]
            ...     attrs = StandardUnitsRegistry.get_unit_attrs(unit_label)

        Use Cases:
            - Dynamic unit validation in configuration parsing
            - Building user interfaces with unit selection
            - Documentation generation
            - Testing completeness of unit handling

        Note:
            The returned tuple is immutable and reflects the current state
            of the StandardUnitsLabels enum. Changes to the enum will be
            automatically reflected in subsequent calls.
        """
        return tuple(StandardUnitsLabels)

"""A module for handling units in the forcing data following the CF conventions."""

from __future__ import annotations

from enum import StrEnum

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

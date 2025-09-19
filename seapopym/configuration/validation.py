"""
Validation functions for configuration parameters.

This module centralizes all validation logic for forcing and parameter units
to avoid code duplication across configuration modules.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pint

if TYPE_CHECKING:
    from numbers import Number

    from pint import Unit

logger = logging.getLogger(__name__)


def verify_forcing_init(value, unit: str | Unit, parameter_name: str):
    """
    Validate and convert a ForcingUnit to the specified unit.

    This function is used to check if the unit of a parameter is correct.
    It raises a DimensionalityError if the unit is not correct.
    """
    # Import here to avoid circular imports
    from seapopym.configuration.no_transport.forcing_parameter import ForcingUnit

    value.convert(unit)
    if isinstance(value, ForcingUnit):
        return value.convert(unit)
    return ForcingUnit(value)


def verify_parameter_init(value: Number, unit: str | pint.Unit, parameter_name: str) -> pint.Quantity:
    """
    Validate and convert a numeric value to a pint.Quantity with the specified unit.

    This function is used to check if the value of a parameter is correct.
    It raises a ValueError if the value is not correct.

    Parameters
    ----------
    value : Number
        The numeric value to validate. Can be a pint.Quantity or raw number.
    unit : str | pint.Unit
        The target unit for validation.
    parameter_name : str
        Name of the parameter being validated (for error messages).

    Returns
    -------
    pint.Quantity
        The validated parameter as a pint.Quantity with correct unit.

    Raises
    ------
    ValueError
        If the unit conversion fails or value is invalid.
    """
    try:
        # Handle case where value already has units
        if hasattr(value, "units"):  # Already a pint.Quantity
            return value.to(unit)
        else:  # Raw numeric value
            return pint.Quantity(value, unit)
    except pint.DimensionalityError as e:
        message = (
            f"Parameter {parameter_name} : {value} is not in the right unit. "
            f"It should be convertible to {unit}. Error: {e}"
        )
        raise ValueError(message) from e
    except Exception as e:
        message = f"Failed to create quantity for parameter {parameter_name} with value {value} and unit {unit}."
        raise ValueError(message) from e

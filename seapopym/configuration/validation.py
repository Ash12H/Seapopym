"""
Validation functions for configuration parameters.

This module centralizes all validation logic for forcing and parameter units
to avoid code duplication across configuration modules.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pint
import xarray as xr

if TYPE_CHECKING:
    from numbers import Number

    from pint import Unit

    from seapopym.configuration.no_transport.forcing_parameter import ForcingUnit

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


class CoordinateCoherenceValidator:
    """Validate coordinate coherence between forcing fields with standardized coordinates."""

    def __init__(self, forcings: dict[str, ForcingUnit]) -> None:
        self.forcings = forcings

    def validate_temporal_coherence(self) -> None:
        """Validate T coordinate coherence between ALL forcings that have T."""
        forcings_with_time = {name: forcing for name, forcing in self.forcings.items() if "T" in forcing.forcing.coords}

        if len(forcings_with_time) < 2:
            return  # Pas assez de forçages temporels pour comparer

        self._validate_coordinate_coherence(forcings_with_time, "T")

    def validate_spatial_coherence(self) -> None:
        """Validate X,Y coordinate coherence between ALL forcings that have spatial dims."""
        # Grouper par combinaisons de coordonnées spatiales
        spatial_groups = {}

        for name, forcing in self.forcings.items():
            spatial_dims = tuple(sorted([dim for dim in ["X", "Y"] if dim in forcing.forcing.coords]))

            if spatial_dims:  # A au moins une coordonnée spatiale
                if spatial_dims not in spatial_groups:
                    spatial_groups[spatial_dims] = {}
                spatial_groups[spatial_dims][name] = forcing

        # Valider la cohérence dans chaque groupe
        for dims, group_forcings in spatial_groups.items():
            if len(group_forcings) > 1:
                self._validate_coordinate_coherence(group_forcings, dims)

    def validate_all_coherence(self) -> None:
        """Validate all possible coherence without assumptions about required fields."""
        # 1. Validation temporelle - tous les forçages avec T
        self.validate_temporal_coherence()

        # 2. Validation spatiale - tous les forçages avec X et/ou Y
        self.validate_spatial_coherence()

        # 3. Si présente, validation Z (optionnelle pour l'avenir)
        forcings_with_z = {name: forcing for name, forcing in self.forcings.items() if "Z" in forcing.forcing.coords}
        if len(forcings_with_z) > 1:
            self._validate_coordinate_coherence(forcings_with_z, "Z")

    def _validate_coordinate_coherence(self, forcings: dict[str, ForcingUnit], coords: str | tuple[str]) -> None:
        """Generic coordinate coherence validation."""
        if isinstance(coords, str):
            coords = (coords,)

        reference_name, reference_forcing = next(iter(forcings.items()))
        reference_coords = {coord: reference_forcing.forcing.coords[coord] for coord in coords}

        for name, forcing in forcings.items():
            if name == reference_name:
                continue

            forcing_coords = {coord: forcing.forcing.coords[coord] for coord in coords}

            if not self._are_coordinates_coherent(reference_coords, forcing_coords):
                coord_desc = "+".join(coords)
                error_details = self._get_coherence_error_details(reference_coords, forcing_coords, coords)
                raise ValueError(
                    f"Coordinate incoherence ({coord_desc}) between '{reference_name}' and '{name}':\n{error_details}"
                )

    def _are_coordinates_coherent(self, coords1: dict[str, xr.DataArray], coords2: dict[str, xr.DataArray]) -> bool:
        """Check if two sets of coordinates are coherent."""
        for coord_name in coords1.keys():
            if coord_name not in coords2:
                return False

            coord1 = coords1[coord_name]
            coord2 = coords2[coord_name]

            # Vérifier que les tailles sont identiques
            if coord1.size != coord2.size:
                return False

            # Pour les coordonnées temporelles, vérifier les valeurs
            if coord_name == "T":
                if not self._are_times_coherent(coord1, coord2):
                    return False

            # Pour les coordonnées spatiales, vérifier les valeurs avec tolérance
            elif coord_name in ["X", "Y", "Z"]:
                if not self._are_spatial_coords_coherent(coord1, coord2):
                    return False

        return True

    def _are_times_coherent(self, time1: xr.DataArray, time2: xr.DataArray) -> bool:
        """Check if two time coordinates are coherent."""
        try:
            # Convertir en timestamps si nécessaire
            if hasattr(time1.values[0], "timestamp"):
                times1 = [t.timestamp() for t in time1.values]
                times2 = [t.timestamp() for t in time2.values]
            else:
                times1 = time1.values
                times2 = time2.values

            return np.array_equal(times1, times2)
        except Exception:
            # Fallback: comparaison directe
            return np.array_equal(time1.values, time2.values)

    def _are_spatial_coords_coherent(self, coord1: xr.DataArray, coord2: xr.DataArray) -> bool:
        """Check if two spatial coordinates are coherent (with tolerance)."""
        try:
            return np.allclose(coord1.values, coord2.values, rtol=1e-5, atol=1e-5)
        except Exception:
            return np.array_equal(coord1.values, coord2.values)

    def _get_coherence_error_details(
        self, coords1: dict[str, xr.DataArray], coords2: dict[str, xr.DataArray], coord_names: tuple[str]
    ) -> str:
        """Generate detailed error message for coordinate incoherence."""
        details = []

        for coord_name in coord_names:
            coord1 = coords1.get(coord_name)
            coord2 = coords2.get(coord_name)

            if coord1 is None or coord2 is None:
                details.append(f"  {coord_name}: Missing coordinate in one of the forcings")
                continue

            if coord1.size != coord2.size:
                details.append(f"  {coord_name}: Different sizes ({coord1.size} vs {coord2.size})")
                continue

            if coord_name == "T":
                # Pour le temps, montrer la plage temporelle
                try:
                    time1_range = f"{coord1.values[0]} to {coord1.values[-1]}"
                    time2_range = f"{coord2.values[0]} to {coord2.values[-1]}"
                    details.append(
                        f"  {coord_name}: Different time ranges\n    Reference: {time1_range}\n    Compared:  {time2_range}"
                    )
                except Exception:
                    details.append(f"  {coord_name}: Different time values")

            elif coord_name in ["X", "Y", "Z"]:
                # Pour l'espace, montrer les bornes et la résolution
                try:
                    min1, max1 = float(coord1.min()), float(coord1.max())
                    min2, max2 = float(coord2.min()), float(coord2.max())
                    resolution1 = float(coord1.diff(coord1.dims[0]).mean()) if coord1.size > 1 else 0
                    resolution2 = float(coord2.diff(coord2.dims[0]).mean()) if coord2.size > 1 else 0

                    details.append(
                        f"  {coord_name}: Different spatial coordinates\n"
                        f"    Reference: {min1:.5f} to {max1:.5f} (res: {resolution1:.5f})\n"
                        f"    Compared:  {min2:.5f} to {max2:.5f} (res: {resolution2:.5f})"
                    )
                except Exception:
                    details.append(f"  {coord_name}: Different spatial values")

        return "\n".join(details) if details else "  Unknown coordinate difference"


def validate_coordinate_coherence(forcings: dict[str, ForcingUnit]) -> None:
    """
    Convenience function to validate coordinate coherence between forcing fields.

    Parameters
    ----------
    forcings : dict[str, ForcingUnit]
        Dictionary of forcing units to validate for coordinate coherence

    Raises
    ------
    ValueError
        If coordinate incoherence is detected between any forcing fields
    """
    validator = CoordinateCoherenceValidator(forcings)
    validator.validate_all_coherence()

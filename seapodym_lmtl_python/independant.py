"""All the functions used to generate or modify the forcings."""

from typing import Iterable

import xarray as xr


def landmask_by_fgroup(
    day_layers: Iterable[int], night_layers: Iterable[int], landmask: xr.DataArray
) -> xr.DataArray:
    """
    The `landmask` has at least 3 dimensions (lat, lon, layer). We are only using the nan cells to generate the
    landmask by functional group.
    """
    pass


def compute_daylength(latitude, longitude) -> xr.DataArray:
    pass


def average_temperature_by_fgroup():
    """Is dependant from compute_daylength and landmask_by_fgroup."""
    pass


def apply_coefficient_to_primary_production():
    """It is equivalent to generate the fisrt cohort of pre-production."""
    pass


def min_temperature_by_cohort():
    pass


def mask_temperature_by_cohort():
    """It uses the min_temperature_by_cohort."""
    pass


def compute_cell_area():
    pass


def compute_mortality_filed():
    pass

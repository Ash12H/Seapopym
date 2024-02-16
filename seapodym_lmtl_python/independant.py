"""All the functions used to generate or modify the forcings."""

from typing import Iterable

from datetime import timedelta

import xarray as xr


def landmask_by_fgroup(
    day_layers: Iterable[int], night_layers: Iterable[int], landmask: xr.DataArray
) -> xr.DataArray:
    """
    The `landmask` has at least 3 dimensions (lat, lon, layer). We are only using the nan cells to generate the
    landmask by functional group.

    landmask can be :
    - temperature
    - user landmask

    Output :
    - landmask [latitude, longitude, functional_group]
    """
    pass


def compute_daylength(
    time: xr.DataArray,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    simulation_timestep: timedelta,
) -> xr.DataArray:
    """
    Use the grid and the time to generate the daylength. Should have a look to a the C++ model -> a python lib is
    available.

    Output:
    - daylength [time{at simulation timestep}, latitude, longitude]

    TODO(Jules): 1. Define the kind of date format to use. datetime, numpy, others ? For which calendar ? Monthly ?
    TODO(Jules): 2. The `simulation_timestep` argument can be automaticly computed inside the initialization process
    TODO(Jules):    (attrs).
    NOTE(Jules): We have to compute the whole time serie because
        - Calendar can be leap aware
        - The timeserie can be shorter than a year
    https://github.com/users/Ash12H/projects/3/views/1?pane=issue&itemId=53453804
    """
    pass


def average_temperature_by_fgroup(
    daylength: xr.DataArray, landmask: xr.DataArray
) -> xr.DataArray:
    """
    Is dependant from compute_daylength and landmask_by_fgroup.

    Input:
    - landmask_by_fgroup()  [time{at simulation timestep}, latitude, longitude]
    - compute_daylength()   [latitude, longitude, functional_group]

    Output:
    - avg_temperature [time{at simulation timestep}, latitude, longitude, functional_group]
    """
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

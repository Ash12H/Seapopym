"""This module contains the function used in the **dependent** process. They are run in sequence in timeseries order."""

from __future__ import annotations

import xarray as xr


def recruitment(
    pre_production: xr.DataArray, recruitment_mask: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Define which part of the pre-production is recruited and return both recuitment and the remaining
    pre-production.

    Input
    -----
    - pre_production [functional_group, time{select(timestep)}, latitude, longitude, cohort_age]
    - recruitment_mask [functional_group, time{select(timestep)}latitude, longitude, cohort_age] :
        coming from the `mask_temperature_by_cohort_by_functional_group()` function.

    Output
    ------
    - recruitment [time{select(timestep)}, latitude, longitude, cohort_age, functional_group]
    - remaining_pre_production [time{select(timestep)}, latitude, longitude, cohort_age, functional_group]

    """


def next_preproduction(
    next_preproduction: xr.DataArray,
    current_unrecruited_preproduction: xr.DataArray,
    max_age: int,
) -> xr.DataArray:
    """
    Return the next pre-production dataarray.

    Input
    -----
    - next_preproduction [functional_group, time{select(next(timestep))}, latitude, longitude, cohort_age]
    - current_unrecruited_preproduction [functional_group, time{select(timestep)}, latitude, longitude, cohort_age]

    Output
    ------
    - next_preproduction [functional_group, time{select(next(timestep))}, latitude, longitude, cohort_age]

    """


def sum_recruitment(recruited: xr.DataArray) -> xr.DataArray:
    """
    Sum the recruitment over the functional_group axis.

    Input
    -----
    - recruited [functional_group, time{select(timestep)}, latitude, longitude, cohort_age]

    Output
    ------
    - recruitment [functional_group, time{select(timestep)}, latitude, longitude]

    """


def compute_biomass(
    recruited: xr.DataArray,
    cell_area: xr.DataArray,
) -> xr.DataArray:
    """
    Compute the recruited biomass.

    Input
    -----
    - recruited [functional_group, time{select(timestep)}, latitude, longitude]
    - cell_area [latitude, longitude]

    Output
    ------
    - biomass [functional_group, time{select(timestep)}, latitude, longitude]
    """

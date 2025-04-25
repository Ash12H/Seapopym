"""A module that provide tools to convert the configuration of the model to a xarray.Dataset."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
import pandas as pd
import xarray as xr

from seapopym.configuration.no_transport import FunctionalGroupUnit
from seapopym.configuration.no_transport.forcing_parameter import ForcingUnit
from seapopym.standard.attributs import functional_group_desc
from seapopym.standard.coordinates import new_cohort
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels

if TYPE_CHECKING:
    from seapopym.configuration.no_transport.forcing_parameter import ForcingParameter


def build_cohort_dataset(functional_group: list[FunctionalGroupUnit]) -> xr.Dataset:
    """
    Return the cohort parameters as a xarray.Dataset. All functional groups doesn't have the same number of
    cohorts.
    """

    def cohort_by_fgroup(fgroup_index: int, timesteps_number: list[int]) -> xr.Dataset:
        """
        Build the cohort axis for a specific functional group using the `timesteps_number` parameter given by the
        user.
        """
        cohort_index = new_cohort(np.arange(0, len(timesteps_number), 1, dtype=int))
        max_timestep = np.cumsum(timesteps_number)
        min_timestep = max_timestep - (np.asarray(timesteps_number) - 1)
        mean_timestep = (max_timestep + min_timestep) / 2

        data_vars = {
            ConfigurationLabels.timesteps_number: (
                (CoordinatesLabels.functional_group, CoordinatesLabels.cohort),
                [timesteps_number],
                {
                    "description": (
                        "The number of timesteps represented in the cohort. If there is no aggregation, all values are "
                        "equal to 1."
                    )
                },
            ),
            ConfigurationLabels.min_timestep: (
                (CoordinatesLabels.functional_group, CoordinatesLabels.cohort),
                [min_timestep],
                {"description": "The minimum timestep index."},
            ),
            ConfigurationLabels.max_timestep: (
                (CoordinatesLabels.functional_group, CoordinatesLabels.cohort),
                [max_timestep],
                {"description": "The maximum timestep index."},
            ),
            ConfigurationLabels.mean_timestep: (
                (CoordinatesLabels.functional_group, CoordinatesLabels.cohort),
                [mean_timestep],
                {"description": "The mean timestep index."},
            ),
        }

        return xr.Dataset(
            coords={CoordinatesLabels.functional_group: [fgroup_index], CoordinatesLabels.cohort: cohort_index},
            data_vars=data_vars,
        )

    all_cohort_timestep = [fgroup.cohort_timestep for fgroup in functional_group]
    all_cohorts_data = [
        cohort_by_fgroup(grp_index, timesteps) for grp_index, timesteps in enumerate(all_cohort_timestep)
    ]
    return xr.merge(all_cohorts_data)


def as_dataset(functional_group: list[FunctionalGroupUnit]) -> xr.Dataset:
    """Return the configuration as a xarray.Dataset."""
    return build_cohort_dataset(functional_group)

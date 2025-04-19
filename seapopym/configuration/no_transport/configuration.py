"""
This module contains the configuration of the model as a xarray.Dataset. It allows to store the model parameters and
the forcings (lazily).
"""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

import numpy as np
from attrs import define, field

from seapopym.configuration.abstract_configuration import AbstractConfiguration
from seapopym.configuration.no_transport.configuration_to_dataset import as_dataset
from seapopym.configuration.no_transport.environment_parameter import EnvironmentParameter
from seapopym.configuration.no_transport.kernel_parameter import KernelParameter
from seapopym.exception.parameter_exception import CohortTimestepConsistencyError

if TYPE_CHECKING:
    from pathlib import Path

    from seapopym.standard.types import SeapopymState


@define
class NoTransportConfiguration(AbstractConfiguration):
    """Configuration for the NoTransportModel."""

    environment: EnvironmentParameter = field(
        factory=EnvironmentParameter, metadata={"description": "The environment parameters for the configuration."}
    )

    kernel: KernelParameter = field(
        factory=KernelParameter, metadata={"description": "The kernel parameters for the configuration."}
    )

    @property
    def state(self: NoTransportConfiguration) -> SeapopymState:
        """The xarray.Dataset that stores the state of the model."""
        # TODO(Jules): Simplify this function
        return as_dataset(
            functional_group=self.functional_group.functional_group,
            forcing_parameter=self.forcing,
        )

    @classmethod
    def parse(cls: NoTransportConfiguration, configuration_file: str | Path | IO) -> NoTransportConfiguration:
        """Parse the configuration file and create a NoTransportConfiguration object."""
        msg = "This method is not implemented yet."
        raise NotImplementedError(msg)

    def __attrs_post_init__(self: NoTransportConfiguration) -> None:
        """
        Check that the timestep of the functional groups is consistent (ie. multiple of) with the timestep of the
        forcings.
        """
        global_timestep = self.forcing.timestep
        for fgroup in self.functional_group.functional_group:
            fgroup_timestep = fgroup.functional_type.cohorts_timesteps
            if not np.all([(ts % global_timestep) == 0 for ts in fgroup_timestep]):
                raise CohortTimestepConsistencyError(
                    cohort_name=fgroup.name, cohort_timesteps=fgroup_timestep, global_timestep=global_timestep
                )

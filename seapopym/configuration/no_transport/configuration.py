"""
This module contains the configuration of the model as a xarray.Dataset. It allows to store the model parameters and
the forcings (lazily).
"""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

import pint
import xarray as xr
from attrs import field, frozen

from seapopym.configuration.no_transport.kernel_parameter import KernelParameter
from seapopym.standard.coordinates import reorder_dims
from seapopym.standard.labels import ConfigurationLabels

if TYPE_CHECKING:
    from pathlib import Path

    from seapopym.configuration.no_transport.forcing_parameter import ForcingParameter
    from seapopym.configuration.no_transport.functional_group_parameter import FunctionalGroupParameter
    from seapopym.standard.types import SeapopymState


@frozen(kw_only=True)
class NoTransportConfiguration:
    """Configuration for the NoTransportModel."""

    forcing: ForcingParameter = field(metadata={"description": "The forcing parameters for the configuration."})
    functional_group: FunctionalGroupParameter = field(
        metadata={"description": "The functional group parameters for the configuration."}
    )

    kernel: KernelParameter = field(
        factory=KernelParameter, metadata={"description": "The kernel parameters for the configuration."}
    )

    @property
    def state(self: NoTransportConfiguration) -> SeapopymState:
        """The xarray.Dataset that stores the state of the model. Data is sent to worker if chunked."""
        data = self.forcing.to_dataset()
        timestep = pint.Quantity(data.cf.indexes["T"].to_series().diff().dt.days.iloc[1], "day")  # Directly as Quantity
        data = xr.merge(
            [
                data,
                self.functional_group.to_dataset(timestep=timestep),
                {ConfigurationLabels.timestep: timestep},
                self.kernel.to_dataset(),
            ],
            combine_attrs="no_conflicts"
        ).pint.dequantify()
        data = reorder_dims(data)
        if self.forcing.parallel:
            data = data.chunk(self.forcing.chunk.as_dict())
        return data.persist()

    @classmethod
    def parse(cls: NoTransportConfiguration, configuration_file: str | Path | IO) -> NoTransportConfiguration:
        """Parse the configuration file and create a NoTransportConfiguration object."""
        msg = "This method is not implemented yet."
        raise NotImplementedError(msg)

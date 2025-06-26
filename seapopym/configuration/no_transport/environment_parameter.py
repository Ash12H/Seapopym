"""A module to manage the environment parameters of the simulation."""

from __future__ import annotations

import logging
from typing import Literal

from attrs import field, frozen, validators

from seapopym.configuration.abstract_configuration import AbstractChunkParameter, AbstractEnvironmentParameter

logger = logging.getLogger(__name__)


@frozen
class ChunkParameter(AbstractChunkParameter):
    """The chunk size of the different dimensions."""

    functional_group: Literal["auto"] | int | None = field(
        default=1,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the functional group dimension."},
    )
    latitude: Literal["auto"] | int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the latitude dimension."},
    )
    longitude: Literal["auto"] | int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the longitude dimension."},
    )
    time: Literal[-1] = field(
        init=False,
        default=-1,
        metadata={
            "description": (
                "The chunk size of the time dimension. "
                "Present only to remind us that time is not divisible due to time dependencies."
            )
        },
    )

    def as_dict(self: ChunkParameter) -> dict:
        """Format to a dictionary as expected by xarray."""
        chunks = {}
        if self.functional_group is not None:
            chunks["functional_group"] = self.functional_group
        if self.latitude is not None:
            chunks["latitude"] = self.latitude
        if self.longitude is not None:
            chunks["longitude"] = self.longitude
        chunks["time"] = self.time
        return chunks


@frozen(kw_only=True)
class EnvironmentParameter(AbstractEnvironmentParameter):
    """Manage the different environment parameters of the simulation."""

    chunk: ChunkParameter = field(
        factory=ChunkParameter,
        validator=validators.instance_of(ChunkParameter),
        metadata={"description": "The chunk size of the different dimensions."},
    )

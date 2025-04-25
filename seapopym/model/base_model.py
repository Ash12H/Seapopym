"""Implementation of the base class for all models."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dask.distributed import Client

    from seapopym.configuration.abstract_configuration import AbstractConfiguration, AbstractEnvironmentParameter
    from seapopym.core.kernel import Kernel
    from seapopym.standard.types import SeapopymState


@dataclass
class BaseModel(abc.ABC):
    """The base class for all models."""

    environment: AbstractEnvironmentParameter
    state: SeapopymState
    kernel: Kernel

    @classmethod
    @abc.abstractmethod
    def from_configuration(cls: type[BaseModel], configuration: AbstractConfiguration) -> BaseModel:
        """Create a model from a configuration."""

    @property
    def client(self: BaseModel) -> Client:
        """The client getter."""
        return self.environment.client.client

    @abc.abstractmethod
    def initialize_dask(self: BaseModel) -> None:
        """
        Initialize local or remote system.

        The client can be a local client or a remote cluster. If the client is not initialized, the model will run
        without parallelism and therefore without dask support.

        ### Xarray documentation on dask :
        Xarray integrates with Dask to support parallel and continuous calculations on datasets that don't fit in
        memory.

        """

    @abc.abstractmethod
    def run(self: BaseModel) -> None:
        """Run the model."""

    @abc.abstractmethod
    def close(self: BaseModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""

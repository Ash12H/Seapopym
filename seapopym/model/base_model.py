"""Implementation of the base class for all models."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seapopym.configuration.abstract_configuration import AbstractConfiguration
    from seapopym.core.kernel import Kernel
    from seapopym.standard.types import SeapopymState


@dataclass
class BaseModel(abc.ABC):
    """The base class for all models."""

    state: SeapopymState
    kernel: Kernel

    @classmethod
    @abc.abstractmethod
    def from_configuration(cls: type[BaseModel], configuration: AbstractConfiguration) -> BaseModel:
        """Create a model from a configuration."""

    @abc.abstractmethod
    def run(self: BaseModel) -> None:
        """Run the model."""

"""Implementation of the base class for all models."""

from __future__ import annotations

import abc
import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from types import TracebackType

    from seapopym.core.kernel import Kernel
    from seapopym.standard.protocols import ConfigurationProtocol
    from seapopym.standard.types import SeapopymState


@dataclass
class BaseModel(abc.ABC):
    """The base class for all models."""

    state: SeapopymState
    kernel: Kernel

    @classmethod
    @abc.abstractmethod
    def from_configuration(cls: type[BaseModel], configuration: ConfigurationProtocol) -> BaseModel:
        """Create a model from a configuration."""

    @abc.abstractmethod
    def run(self: BaseModel) -> None:
        """Run the model."""

    def __enter__(self: BaseModel) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self: BaseModel,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit context manager and cleanup memory."""
        # Clean up large objects
        if hasattr(self, "state"):
            del self.state
        if hasattr(self, "kernel"):
            del self.kernel

        # Force garbage collection for genetic algorithms usage
        gc.collect()

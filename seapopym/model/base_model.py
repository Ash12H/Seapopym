"""Implementation of the base class for all models."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from types import TracebackType

    from seapopym.core.kernel import Kernel
    from seapopym.standard.protocols import ConfigurationProtocol, ModelProtocol
    from seapopym.standard.types import SeapopymState


@dataclass
class BaseModel:
    """The base class for all models.

    Implements ModelProtocol via duck typing.
    """

    state: SeapopymState
    kernel: Kernel

    @classmethod
    def from_configuration(cls: type[BaseModel], configuration: ConfigurationProtocol) -> BaseModel:
        """Create a model from a configuration.

        Must be implemented by subclasses.
        """
        msg = f"Subclass {cls.__name__} must implement from_configuration method"
        raise NotImplementedError(msg)

    def run(self: BaseModel) -> None:
        """Run the model.

        Must be implemented by subclasses.
        """
        msg = f"Subclass {self.__class__.__name__} must implement run method"
        raise NotImplementedError(msg)

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

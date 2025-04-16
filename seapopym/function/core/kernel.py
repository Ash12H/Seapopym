"""Define the Kernels used in the model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ParamSpecArgs, ParamSpecKwargs

import xarray as xr
from attrs import define, field

from seapopym.function.core.template import BaseTemplate, ForcingTemplate, StateTemplate

# from seapopym.logging.custom_logger import logger
from seapopym.standard.types import SeapopymForcing, SeapopymState

if TYPE_CHECKING:
    from collections.abc import Iterable

kernel_registry = {}
"""
The kernel registry is a dictionary that stores the KernelUnits used in the models. Any function decorated with the
@registry_kernel decorator will be added to this registry. The keys are the names of the KernelUnits and the values
are the KernelUnits objects that contain the function and its template. Use the registry to create new models.
"""


def registry_kernel(name: str, template: BaseTemplate):
    """Decorator to register a KernelUnits function."""

    def decorator(func):
        """Register the function as a KernelUnits function."""
        kernel_registry[name] = KernelUnits(name=name, template=template, function=func)
        return func

    return decorator


@define
class KernelUnits:
    """
    The KernelUnits class is used to define a kernel function that can be applied to the model state.
    It contains the function, its template, and the arguments to be passed to the function.
    """

    name: str
    template: BaseTemplate
    function: Callable[[SeapopymState, ParamSpecArgs, ParamSpecKwargs], SeapopymState | SeapopymForcing]
    # TODO(jules): Remove args and kwargs. They will be stored in the state of the model.
    args: ParamSpecArgs = field(factory=tuple)
    kwargs: ParamSpecKwargs = field(factory=dict)

    def _map_block_without_dask(self: KernelUnits, state: SeapopymState):
        # logger.debug(f"Direct computation for {self.function.__name__}.")
        results = self.function(state, *self.args, **self.kwargs)

        if isinstance(self.template, ForcingTemplate):
            if isinstance(results, SeapopymState):
                msg = "When the function returns a xarray.Dataset, the template attribut should be a ForcingTemplate."
                raise TypeError(msg)
            results.name = self.template.name
            return results.assign_attrs(self.template.attrs)

        if isinstance(self.template, StateTemplate):
            if isinstance(results, SeapopymForcing):
                msg = "When the function returns a xarray.Dataset, the template attribut should be a StateTemplate."
                raise TypeError(msg)
            for template in self.template.template:
                if template.name not in results:
                    msg = f"Variable {template.name} is not in the results."
                    raise ValueError(msg)
                results[template.name] = results[template.name].assign_attrs(template.attrs)
            return results

        msg = "The template attribut should be a ForcingTemplate or a StateTemplate."
        raise TypeError(msg)

    def _map_block_with_dask(self: KernelUnits, state: SeapopymState) -> SeapopymForcing | SeapopymState:
        # logger.debug(f"Creating template for {self.function.__name__}.")

        result_template = self.template.generate(state)

        # logger.debug(f"Applying map_blocks to {self.function.__name__}.")
        return xr.map_blocks(self.function, state, template=result_template, args=self.args, kwargs=self.kwargs)

    def run(self: KernelUnits, state: SeapopymState) -> SeapopymState | SeapopymForcing:
        if len(state.chunks) == 0:
            return self._map_block_without_dask(state)

        return self._map_block_with_dask(state)


class BaseKernel:
    """
    The BaseKernel class is used to define a kernel that can be applied to the model state.
    It contains a list of KernelUnits that will be applied in order.
    """

    def __init__(self: BaseKernel, functions: Iterable[KernelUnits], chunk: dict[str, int]) -> None:
        self._functions = functions
        self._chunk = chunk

    def run(self: BaseKernel, state: SeapopymState) -> SeapopymState:
        """Run all functions in the kernel in order."""
        for kernel in self._kernels:
            results = kernel.run(state)
            state = state.merge(results)
        return state

    def template(self: BaseKernel, state: SeapopymState) -> SeapopymState:
        """
        Generate an empty Dataset that represent the state of the model at the end of execution. Usefull for
        size estimation.
        """
        return xr.merge([unit.template.generate(state) for unit in self._functions] + [state])


def kernel_factory(class_name: str, functions: list[str]) -> BaseKernel:
    """
    Create a custom kernel class with the specified name and functions.

    Parameters
    ----------
    class_name : str
        The name to assign to the custom kernel class.
    functions : list of str
        A list of function names to be used in the kernel. These functions
        must be registered in the `kernel_registry`.

    Returns
    -------
    BaseKernel
        A dynamically created kernel class with the specified name and
        functions.

    Notes
    -----
    The returned class inherits from `BaseKernel` and is initialized with
    the provided functions and a chunk dictionary.

    """

    class CustomKernel(BaseKernel):
        def __init__(self, chunk: dict):
            super().__init__(functions, chunk)

    CustomKernel.__name__ = class_name
    return CustomKernel

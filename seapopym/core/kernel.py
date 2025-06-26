"""Define the Kernels used in the model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import xarray as xr

from seapopym.core.template import Template, TemplateUnit

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from seapopym.standard.types import SeapopymForcing, SeapopymState

logger = logging.getLogger(__name__)


@dataclass
class KernelUnit:
    """
    The KernelUnit class is used to define a kernel function that can be applied to the model state.
    It contains the function, its template, and the arguments to be passed to the function.
    """

    name: str
    template: Template
    function: Callable[[SeapopymState], xr.Dataset]
    # TODO(Jules): Add possibility to remove temporary variables after function ?
    to_remove_from_state: list[str] | None = None

    def _map_block_without_dask(self: KernelUnit, state: SeapopymState) -> xr.Dataset:
        results = self.function(state)
        for template in self.template.template_unit:
            if template.name not in results:
                msg = f"Variable {template.name} is not in the results."
                raise ValueError(msg)
            results[template.name] = results[template.name].assign_attrs(template.attrs)
        return results

    def _map_block_with_dask(self: KernelUnit, state: SeapopymState) -> xr.Dataset:
        result_template = self.template.generate(state)
        return xr.map_blocks(self.function, state, template=result_template)

    def run(self: KernelUnit, state: SeapopymState) -> SeapopymState | SeapopymForcing:
        """Execute the kernel function on the model state and return the results as Dataset."""
        if len(state.chunks) == 0:
            return self._map_block_without_dask(state)

        return self._map_block_with_dask(state)


def kernel_unit_factory(
    name: str,
    function: Callable[[SeapopymState], xr.Dataset],
    template: Iterable[type[TemplateUnit]],
    to_remove_from_state: list[str] | None = None,
) -> type[KernelUnit]:
    """
    Create a custom kernel unit class with the specified name and function.

    Parameters
    ----------
    name : str
        The name to assign to the custom kernel unit class.
    function : Callable
        The function to be used in the kernel unit. It should accept a SeapopymState
        and return a Dataset.
    template : list of TemplateUnit
        A list of TemplateUnit classes to be used in the kernel unit. These TemplateUnits
        must be registered in the `template_unit_registry`. Be aware that **the
        order of the list matters**, as the template units will be applied in
        the order they are listed.
    to_remove_from_state : list of str, optional
        A list of variable names to be removed from the state after the kernel unit
        has been executed. If not provided, no variables will be removed.

    Returns
    -------
    KernelUnit
        A dynamically created kernel unit class with the specified name and
        function.

    Notes
    -----
    The returned class inherits from `KernelUnit` and is initialized with
    the provided function and a chunk dictionary.

    """

    class CustomKernelUnit(KernelUnit):
        def __init__(self, chunk: dict[str, int]) -> None:
            super().__init__(
                name=name,
                function=function,
                template=Template(template_unit=[template_class(chunk) for template_class in template]),
                to_remove_from_state=to_remove_from_state,
            )

    CustomKernelUnit.__name__ = name

    return CustomKernelUnit


class Kernel:
    """
    The Kernel class is used to define a kernel that can be applied to the model state.
    It contains a list of KernelUnit that will be applied in order.
    """

    def __init__(self: Kernel, kernel_unit: Iterable[type[KernelUnit]], chunk: dict[str, int]) -> None:
        self.kernel_unit = [ku(chunk) for ku in kernel_unit]

    def run(self: Kernel, state: SeapopymState) -> SeapopymState:
        """Run all kernel_unit in the kernel in order."""
        for ku in self.kernel_unit:
            results = ku.run(state)
            state = results.merge(state, compat="override")
            for var in ku.to_remove_from_state or []:
                if var in state:
                    state = state.drop_vars(var)
        return state

    def template(self: Kernel, state: SeapopymState) -> SeapopymState:
        """
        Generate an empty Dataset that represent the state of the model at the end of execution. Usefull for
        size estimation.
        """
        return xr.merge([state] + [unit.template.generate(state) for unit in self.kernel_unit], compat="override")


def kernel_factory(class_name: str, kernel_unit: list[type[KernelUnit]]) -> Kernel:
    """
    Create a custom kernel class with the specified name and functions.

    Parameters
    ----------
    class_name : str
        The name to assign to the custom kernel class.
    kernel_unit : list of str
        A list of KernelUnit names to be used in the kernel. These KernelUnits
        must be registered in the `kernel_unit_registry`. Be aware that **the
        order of the list matters**, as the kernel units will be applied in
        the order they are listed.

    Returns
    -------
    Kernel
        A dynamically created kernel class with the specified name and
        kernel units.

    Notes
    -----
    The returned class inherits from `Kernel` and is initialized with
    the provided functions and a chunk dictionary.

    """

    class CustomKernel(Kernel):
        def __init__(self, chunk: dict):
            super().__init__(kernel_unit=kernel_unit, chunk=chunk)

    CustomKernel.__name__ = class_name
    return CustomKernel

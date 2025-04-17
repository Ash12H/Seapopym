"""Define the Kernels used in the model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ParamSpecArgs, ParamSpecKwargs

import xarray as xr
from attrs import define, field

from seapopym.function.core.template import TemplateUnit, Template

# from seapopym.logging.custom_logger import logger
from seapopym.standard.types import SeapopymForcing, SeapopymState

if TYPE_CHECKING:
    from collections.abc import Iterable

kernel_unit_registry = {}
"""
The kernel registry is a dictionary that stores the KernelUnit used in the models. Any function decorated with the
@registry_kernel decorator will be added to this registry. The keys are the names of the KernelUnit and the values
are the KernelUnit objects that contain the function and its template. Use the registry to create new models.
"""


# TODO(Jules): Use dataclass instead of attrs
@define
class KernelUnit:
    """
    The KernelUnit class is used to define a kernel function that can be applied to the model state.
    It contains the function, its template, and the arguments to be passed to the function.
    """

    name: str
    template: Template
    function: Callable[[SeapopymState, ParamSpecArgs, ParamSpecKwargs], xr.Dataset]
    # TODO(jules): Remove args and kwargs. They will be stored in the state of the model.
    args: ParamSpecArgs = field(factory=tuple)
    kwargs: ParamSpecKwargs = field(factory=dict)

    def _map_block_without_dask(self: KernelUnit, state: SeapopymState) -> xr.Dataset:
        # logger.debug(f"Direct computation for {self.function.__name__}.")
        results = self.function(state, *self.args, **self.kwargs)
        # TODO(Jules): Remove print funcitons
        print("template", self.template)
        for template in self.template.template_unit:
            print("template.name", template.name)
            if template.name not in results:
                msg = f"Variable {template.name} is not in the results."
                raise ValueError(msg)
            results[template.name] = results[template.name].assign_attrs(template.attrs)
        return results

    def _map_block_with_dask(self: KernelUnit, state: SeapopymState) -> xr.Dataset:
        # logger.debug(f"Creating template for {self.function.__name__}.")
        result_template = self.template.generate(state)
        # logger.debug(f"Applying map_blocks to {self.function.__name__}.")
        return xr.map_blocks(self.function, state, template=result_template, args=self.args, kwargs=self.kwargs)

    def run(self: KernelUnit, state: SeapopymState) -> SeapopymState | SeapopymForcing:
        """Execute the kernel function on the model state and return the results as Dataset."""
        if len(state.chunks) == 0:
            return self._map_block_without_dask(state)

        return self._map_block_with_dask(state)


def kernel_unit_registry_factory(name: str, template: Iterable[TemplateUnit]):
    """Decorator to register a KernelUnit function."""

    def decorator(func):
        """Register the function as a KernelUnit function."""

        class CustomKernelUnit(KernelUnit):
            def __init__(self, chunk: dict[str, int]) -> None:
                super().__init__(
                    name=name,
                    template=Template([t for t in template]),
                    function=func,
                )

        CustomKernelUnit.__name__ = name

        kernel_unit_registry[name] = CustomKernelUnit

        return func

    return decorator


class BaseKernel:
    """
    The BaseKernel class is used to define a kernel that can be applied to the model state.
    It contains a list of KernelUnit that will be applied in order.
    """

    def __init__(self: BaseKernel, kernel_unit: Iterable[KernelUnit], chunk: dict[str, int]) -> None:
        self._kernel_unit = [ku(chunk) for ku in kernel_unit]
        self._chunk = chunk

    def run(self: BaseKernel, state: SeapopymState) -> SeapopymState:
        """Run all kernel_unit in the kernel in order."""
        for kernel in self._kernel_unit:
            results = kernel.run(state)
            # TODO(Jules): The `results` might override variables in the state. If so, we might use another method.
            # For example we can simply use state[var] = results[var] instead of merge.
            state = state.merge(results)
        return state

    def template(self: BaseKernel, state: SeapopymState) -> SeapopymState:
        """
        Generate an empty Dataset that represent the state of the model at the end of execution. Usefull for
        size estimation.
        """
        return xr.merge([unit.template.generate(state) for unit in self._kernel_unit] + [state])


def kernel_factory(class_name: str, kernel_unit: list[str]) -> BaseKernel:
    """
    Create a custom kernel class with the specified name and functions.

    Parameters
    ----------
    class_name : str
        The name to assign to the custom kernel class.
    kernel_unit : list of str
        A list of KernelUnit names to be used in the kernel. These KernelUnits
        must be registered in the `kernel_unit_registry`.

    Returns
    -------
    BaseKernel
        A dynamically created kernel class with the specified name and
        kernel units.

    Notes
    -----
    The returned class inherits from `BaseKernel` and is initialized with
    the provided functions and a chunk dictionary.

    """

    class CustomKernel(BaseKernel):
        def __init__(self, chunk: dict):
            super().__init__(kernel_unit=[kernel_unit_registry[ku_name] for ku_name in kernel_unit], chunk=chunk)

    CustomKernel.__name__ = class_name
    return CustomKernel


KernelNoTransport = kernel_factory(
    class_name="KernelNoTransport",
    kernel_unit=[
        "global_mask",
        "mask_by_fgroup",
        "day_length",
        "average_temperature",
        "primary_production_by_fgroup",
        "min_temperature_by_cohort",
        "mask_temperature",
        "cell_area",
        "mortality_field",
        "production",
        "biomass",
    ],
)

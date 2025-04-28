"""
Functions used to generalize the usage of map_blocks function in the model.

Notes
-----
This module is used to generate a template for a new variable that can be used in a xarray.map_blocks function. The
template is based on the state of the model and the dimensions of the new variable. The template can be chunked if
needed.

### xarray documentation:

> If none of the variables in obj is backed by dask arrays, calling this function is equivalent to calling
> `func(obj, *args, **kwargs)`.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import dask.array as da
import xarray as xr
from attr import field, validators
from attrs import frozen

from seapopym.standard import coordinates
from seapopym.standard.types import ForcingName, SeapopymDims, SeapopymForcing

if TYPE_CHECKING:
    from seapopym.standard.types import ForcingAttrs, SeapopymState


@frozen(kw_only=True)
class BaseTemplate(ABC):
    @abstractmethod
    def generate(self: BaseTemplate, state: SeapopymState) -> SeapopymForcing | SeapopymState:
        """Generate an empty xr.DataArray/Dataset."""


@frozen(kw_only=True)
class TemplateUnit(BaseTemplate):
    name: ForcingName
    attrs: ForcingAttrs
    dims: Iterable[SeapopymDims | SeapopymForcing] = field(validator=validators.instance_of(Iterable))
    chunks: dict[str, int] | None = None
    dtype: type | None = field(default=None, validator=validators.optional(validators.instance_of(type)))

    @dims.validator
    def _validate_dims(self, attribute, value) -> None:
        """Check if the dimensions are either SeapopymDims or SeapopymForcing objects."""
        for dim in self.dims:
            if not isinstance(dim, SeapopymDims | SeapopymForcing):
                msg = f"Dimension {dim} must be either a SeapopymDims or SeapopymForcing object."
                raise TypeError(msg)

    def generate(self: TemplateUnit, state: SeapopymState) -> SeapopymForcing:
        for dim in self.dims:
            if isinstance(dim, SeapopymDims) and state is None:
                msg = "You need to provide the state of the model to generate the template."
                raise ValueError(msg)
            if isinstance(dim, SeapopymDims) and dim not in state.cf.coords:
                msg = f"Dimension {dim} is not defined in the state of the model."
                raise ValueError(msg)

        coords = [dim if isinstance(dim, SeapopymForcing) else state.cf[dim] for dim in self.dims]
        coords_size = [dim.size for dim in coords]
        coords_name = [dim.name for dim in coords]
        if self.chunks is not None:
            unordered_chunks = {state.cf[k].name: v for k, v in self.chunks.items()}
            ordered_chunks = [unordered_chunks.get(dim.name, None) for dim in coords]
        else:
            ordered_chunks = {}

        # NOTE(Jules): dask empty array initialization is faster than numpy version
        template = xr.DataArray(
            da.empty(coords_size, chunks=ordered_chunks, dtype=self.dtype),
            coords=coords,
            dims=coords_name,
            name=self.name,
            attrs=self.attrs,
        )
        return coordinates.CoordinatesLabels.order_data(template)


def template_unit_factory(
    name: ForcingName,
    attributs: ForcingAttrs,
    dims: Iterable[SeapopymDims | SeapopymForcing],
    dtype: type | None = None,
) -> type[BaseTemplate]:
    class CustomTemplateUnit(TemplateUnit):
        def __init__(self, chunk: dict):
            super().__init__(name=name, attrs=attributs, dims=dims, chunks=chunk, dtype=dtype)

    CustomTemplateUnit.__name__ = name
    return CustomTemplateUnit


@frozen(kw_only=True)
class Template(BaseTemplate):
    template_unit: Iterable[TemplateUnit]

    def generate(self: Template, state: SeapopymState) -> SeapopymState:
        results = {template.name: template.generate(state) for template in self.template_unit}
        return xr.Dataset(results)

"""Funcitons used to generate a landmask from any forcing data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401  # noqa: F401
import xarray as xr

from seapopym.core import kernel, template
from seapopym.standard.attributs import global_mask_desc
from seapopym.standard.labels import CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymState


def global_mask(state: SeapopymState) -> xr.Dataset:
    """Create a global mask from temperature forcing in the state of the model."""
    mask = state[ForcingLabels.temperature].cf.isel(T=0).notnull().cf.reset_coords("T", drop=True)
    return xr.Dataset({ForcingLabels.global_mask: mask})


GlobalMaskTemplate = template.template_unit_factory(
    name=ForcingLabels.global_mask,
    attributs=global_mask_desc,
    dims=[CoordinatesLabels.Y, CoordinatesLabels.X, CoordinatesLabels.Z],
    dtype=bool,
)


GlobalMaskKernel = kernel.kernel_unit_factory(name="global_mask", template=[GlobalMaskTemplate], function=global_mask)

"""Wrapper for the application of the transfert cooeficient to primary production. Use xarray.map_block."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.core import kernel, template
from seapopym.standard.attributs import apply_coefficient_to_primary_production_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymState


def primary_production_by_fgroup(state: SeapopymState) -> xr.Dataset:
    """
    It is equivalent to generate the fisrt cohort of pre-production.

    Input
    -----
    - primary_production [time, latitude, longitude]
    - functional_group_coefficient [functional_group]

    Output
    ------
    - primary_production [functional_group, time, latitude, longitude]
    """
    primary_production = state[ForcingLabels.primary_production]
    pp_by_fgroup_gen = (i * primary_production for i in state[ConfigurationLabels.energy_transfert])
    pp_by_fgroup_gen = xr.concat(pp_by_fgroup_gen, dim=CoordinatesLabels.functional_group)
    return xr.Dataset({ForcingLabels.primary_production_by_fgroup: pp_by_fgroup_gen})


PrimaryProductionByFgroupTemplate = template.template_unit_factory(
    name=ForcingLabels.primary_production_by_fgroup,
    attributs=apply_coefficient_to_primary_production_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
)


PrimaryProductionByFgroupKernel = kernel.kernel_unit_factory(
    name="primary_production_by_fgroup",
    template=[PrimaryProductionByFgroupTemplate],
    function=primary_production_by_fgroup,
)

PrimaryProductionByFgroupKernelLight = kernel.kernel_unit_factory(
    name="primary_production_by_fgroup_light",
    template=[PrimaryProductionByFgroupTemplate],
    function=primary_production_by_fgroup,
    to_remove_from_state=[ForcingLabels.primary_production],
)

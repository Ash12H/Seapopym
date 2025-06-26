"""
This module contains the post-production function used to compute the biomass.
They are run after the production process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.core import kernel, template
from seapopym.function.compiled_functions.biomass_compiled_functions import biomass_sequence
from seapopym.standard.attributs import biomass_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymForcing, SeapopymState


def biomass(state: SeapopymState) -> xr.Dataset:
    """Wrap the biomass computation arround the Numba function `biomass_sequence`."""

    def _format_fields(forcing: SeapopymForcing) -> SeapopymForcing:
        """Format the fields to be used in the biomass computation."""
        return np.nan_to_num(forcing.data, 0.0).astype(np.float64)

    state = CoordinatesLabels.order_data(state)
    recruited = _format_fields(state[ForcingLabels.recruited])
    mortality = _format_fields(state[ForcingLabels.mortality_field])
    if ConfigurationLabels.initial_condition_biomass in state:
        initial_conditions = _format_fields(state[ConfigurationLabels.initial_condition_biomass])
    else:
        initial_conditions = None
    biomass = biomass_sequence(recruited=recruited, mortality=mortality, initial_conditions=initial_conditions)
    biomass = xr.DataArray(
        dims=state[ForcingLabels.mortality_field].dims,
        coords=state[ForcingLabels.mortality_field].coords,
        data=biomass,
    )
    return xr.Dataset({ForcingLabels.biomass: biomass})


BiomassTemplate = template.template_unit_factory(
    name=ForcingLabels.biomass,
    attributs=biomass_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
)


BiomassKernel = kernel.kernel_unit_factory(name="biomass", template=[BiomassTemplate], function=biomass)

BiomassKernelLight = kernel.kernel_unit_factory(
    name="biomass_light",
    template=[BiomassTemplate],
    function=biomass,
    to_remove_from_state=[ForcingLabels.recruited, ForcingLabels.mortality_field],
)

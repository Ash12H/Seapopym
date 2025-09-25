"""Apply the survival rate to the recruited biomass."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from seapopym.core import kernel, template
from seapopym.standard.attributs import recruited_desc
from seapopym.standard.labels import CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymState


def apply_survival_rate_to_recruitment(state: SeapopymState) -> xr.Dataset:
    """
    Apply survival rate to recruited biomass.

    Multiply the recruited biomass by the survival rate to account for
    mortality effects from ocean acidification and temperature.

    Parameters
    ----------
    state : SeapopymState
        The model state containing recruited biomass and survival rate data.

    Returns
    -------
    xr.Dataset
        Dataset containing the adjusted recruited biomass.

    Depends on
    ----------
    - recruited (from production functions)
    - survival_rate (from survival_rate_bednarsek function)
    """
    recruited = state[ForcingLabels.recruited]
    survival_rate = state[ForcingLabels.survival_rate]

    recruited_adjusted = recruited * survival_rate

    return xr.Dataset({ForcingLabels.recruited: recruited_adjusted})


RecruitedAdjustedTemplate = template.template_unit_factory(
    name=ForcingLabels.recruited,
    attributs=recruited_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
)

ApplySurvivalRateToRecruitmentKernel = kernel.kernel_unit_factory(
    name="apply_survival_rate_to_recruitment",
    template=[RecruitedAdjustedTemplate],
    function=apply_survival_rate_to_recruitment,
)

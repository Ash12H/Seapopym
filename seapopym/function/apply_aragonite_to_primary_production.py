""""""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import xarray as xr
import numpy as np

from seapopym.core import kernel, template
from seapopym.standard.attributs import apply_coefficient_to_primary_production_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymState


def sigmoid(x,slope,center):
    """    
    Parameters
    ----------
    x : float or array-like
        Input value(s).
    slope : float
        Steepness of the sigmoid. Positive filters low values, negative filters high values.
    center : float
        The center (inflection point) of the sigmoid.
    
    Returns
    -------
    float or array-like
        Sigmoid-transformed value(s).
    
    Notes
    -----
    If slope > 0: values below center tend toward 0, above tend toward 1.
    If slope < 0: values above center tend toward 0, below tend toward 1.
    """
    return 1 / (1 + np.exp(-slope * (x - center)))

def primary_production_by_fgroup(state: SeapopymState) -> xr.Dataset:
    """
    It is equivalent to generate the fisrt cohort of pre-production.

    Input
    -----
    - primary_production [time, latitude, longitude]
    - avg acidity (aragonite) [functional_group,time, latitude, longitude] (for future version if several layers)
    - functional_group_coefficient [functional_group]
    - gamma_sigmoid_aragonite (slope of the sigmoid)
    - center_sigmoid_aragonite (center of the sigmoid)

    Output
    ------
    - primary_production [functional_group, time, latitude, longitude]
    """
    primary_production = state[ForcingLabels.primary_production]
    average_acidity = state[ForcingLabels.avg_acidity_by_fgroup]
    gamma_sigmoid_aragonite = state[ConfigurationLabels.gamma_sigmoid_aragonite]
    center_sigmoid_aragonite = state[ConfigurationLabels.center_sigmoid_aragonite]
    pp_by_fgroup_gen=[]
    for fgroup in state[CoordinatesLabels.functional_group]:
        coef = state[ConfigurationLabels.energy_transfert].sel({CoordinatesLabels.functional_group:fgroup})
        sig = sigmoid(
            average_acidity.sel({CoordinatesLabels.functional_group:fgroup}),
            gamma_sigmoid_aragonite,
            center_sigmoid_aragonite
        )
        pp = coef * primary_production * sig
        pp_by_fgroup_gen.append(pp)
    pp_by_fgroup_gen = xr.concat(pp_by_fgroup_gen, dim=CoordinatesLabels.functional_group)
    return xr.Dataset({ForcingLabels.primary_production_by_fgroup: pp_by_fgroup_gen})


PrimaryProductionByFgroupTemplate = template.template_unit_factory(
    name=ForcingLabels.primary_production_by_fgroup,
    attributs=apply_coefficient_to_primary_production_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
)


PrimaryProductionByFgroupKernelAragonite = kernel.kernel_unit_factory(
    name="primary_production_by_fgroup",
    template=[PrimaryProductionByFgroupTemplate],
    function=primary_production_by_fgroup,
)
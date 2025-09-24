"""Configuration for acidity model with Bednarsek mortality equation."""

from attrs import field, frozen

from seapopym.configuration import acidity
from seapopym.configuration.acidity_bed.functional_group_parameter import FunctionalGroupParameter


@frozen(kw_only=True)
class AcidityBedConfiguration(acidity.AcidityConfiguration):
    """Configuration for the acidity model using Bednarsek mortality equation."""

    functional_group: FunctionalGroupParameter = field(
        metadata={"description": "The functional group parameters for the Bednarsek acidity configuration."}
    )

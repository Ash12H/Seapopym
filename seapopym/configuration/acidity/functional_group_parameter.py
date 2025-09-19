from functools import partial

import pint
from attrs import field, frozen, validators

from seapopym.configuration import no_transport
from seapopym.configuration.validation import verify_parameter_init
from seapopym.standard.labels import ConfigurationLabels


@frozen(kw_only=True)
class FunctionalTypeParameter(no_transport.FunctionalTypeParameter):
    """
    Adapted from the original FunctionalTypeParameter class to include parameters related to the effect of pH on
    mortality.
    """

    lambda_acidity_0: pint.Quantity = field(
        alias=ConfigurationLabels.lambda_acidity_0,
        converter=partial(verify_parameter_init, unit="1/day", parameter_name=ConfigurationLabels.lambda_acidity_0),
        validator=validators.gt(0),
        metadata={"description": "Value of lambda_acidity when pH is 0."},
    )
    gamma_lambda_acidity: pint.Quantity = field(
        alias=ConfigurationLabels.gamma_lambda_acidity,
        converter=partial(verify_parameter_init, unit="dimensionless", parameter_name=ConfigurationLabels.gamma_lambda_acidity),
        metadata={"description": "Rate of the mortality due to acidity (pH)."},
    )


@frozen(kw_only=True)
class FunctionalGroupUnit(no_transport.FunctionalGroupUnit):
    """Represent a functional group."""

    functional_type: FunctionalTypeParameter = field(
        validator=validators.instance_of(FunctionalTypeParameter),
        metadata={"description": "Parameters linked to the relation between temperature and the functional group."},
    )


@frozen(kw_only=True)
class FunctionalGroupParameter(no_transport.FunctionalGroupParameter):
    """This data class is used to store the parameters of all functional groups."""

    functional_group: list[FunctionalGroupUnit] = field(metadata={"description": "List of all functional groups."})

"""Functional group parameters for acidity model with Bednarsek mortality equation."""

from functools import partial

import pint
from attrs import field, frozen, validators

from seapopym.configuration import no_transport
from seapopym.configuration.validation import verify_parameter_init
from seapopym.standard.labels import ConfigurationLabels


@frozen(kw_only=True)
class FunctionalTypeParameter(no_transport.FunctionalTypeParameter):
    """
    Functional type parameters extended with Bednarsek mortality parameters.

    Adds parameters for the Bednarsek et al. (2022) mortality equation:
    mortality = lambda_0_bed + gamma_lambda_temperature_bed * T + gamma_lambda_acidity_bed * pH
    """

    lambda_0_bed: pint.Quantity = field(
        alias=ConfigurationLabels.lambda_0_bed,
        converter=partial(verify_parameter_init, unit="dimensionless", parameter_name=ConfigurationLabels.lambda_0_bed),
        metadata={"description": "Value of lambda when temperature is 0°C and aragonite is 0."},
    )
    gamma_lambda_temperature_bed: pint.Quantity = field(
        alias=ConfigurationLabels.gamma_lambda_temperature_bed,
        converter=partial(
            verify_parameter_init, unit="1/degC", parameter_name=ConfigurationLabels.gamma_lambda_temperature_bed
        ),
        metadata={"description": "Sensitivity to temperature in Bednarsek equation."},
    )
    gamma_lambda_acidity_bed: pint.Quantity = field(
        alias=ConfigurationLabels.gamma_lambda_acidity_bed,
        converter=partial(
            verify_parameter_init, unit="dimensionless", parameter_name=ConfigurationLabels.gamma_lambda_acidity_bed
        ),
        metadata={"description": "Sensitivity to aragonite in Bednarsek equation."},
    )


@frozen(kw_only=True)
class FunctionalGroupUnit(no_transport.FunctionalGroupUnit):
    """Represent a functional group with Bednarsek parameters."""

    functional_type: FunctionalTypeParameter = field(
        validator=validators.instance_of(FunctionalTypeParameter),
        metadata={"description": "Parameters for temperature and acidity relationships using Bednarsek equation."},
    )


@frozen(kw_only=True)
class FunctionalGroupParameter(no_transport.FunctionalGroupParameter):
    """Store parameters for all functional groups using Bednarsek mortality equation."""

    functional_group: list[FunctionalGroupUnit] = field(
        metadata={"description": "List of all functional groups with Bednarsek parameters."}
    )

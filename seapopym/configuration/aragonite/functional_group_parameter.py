from numbers import Number

from attrs import field, frozen, validators

from seapopym.configuration import no_transport


@frozen(kw_only=True)
class FunctionalTypeParameter(no_transport.FunctionalTypeParameter):
    """
    Adapted from the original FunctionalTypeParameter class to 
    include parameters related to the effect of aragonite on the juneviles
    """

    center_sigmoid_aragonite: Number = field(
        validator=[validators.gt(0)],
        converter=float,
        metadata={"description": "Center of the sigmoid"},
    )
    gamma_sigmoid_aragonite: Number = field(
        converter=float, metadata={"description": "Rate of the sigmoid."}
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
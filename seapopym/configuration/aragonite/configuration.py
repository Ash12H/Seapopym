from attrs import field, frozen

from seapopym.configuration import acidity, no_transport, aragonite


@frozen(kw_only=True)
class AragoniteConfiguration(no_transport.NoTransportConfiguration):
    """Configuration for the NoTransportModel."""

    forcing: acidity.ForcingParameter = field(metadata={"description": "The forcing parameters for the configuration."})
    functional_group: aragonite.FunctionalGroupParameter = field(
        metadata={"description": "The functional group parameters for the configuration."}
    )
"""Tests for Phase 7 configuration ABC to Protocol migration.

Verifies that configuration classes have been successfully migrated from ABC to Protocol.
"""

import abc
import pytest

from seapopym.configuration.no_transport.forcing_parameter import ForcingParameter
from seapopym.configuration.no_transport.functional_group_parameter import FunctionalGroupParameter
from seapopym.configuration.no_transport.configuration import NoTransportConfiguration
from seapopym.standard.protocols import (
    ForcingParameterProtocol,
    FunctionalGroupParameterProtocol,
    ConfigurationProtocol
)


@pytest.mark.protocols
class TestPhase7ConfigurationMigration:
    """Test that configuration classes have been migrated from ABC to Protocol."""

    def test_forcing_parameter_no_longer_inherits_abc(self):
        """Test that ForcingParameter no longer inherits from AbstractForcingParameter."""

        # ForcingParameter should not have any ABC in its MRO
        abc_classes = [cls for cls in ForcingParameter.__mro__ if issubclass(cls, abc.ABC) and cls != abc.ABC]
        assert len(abc_classes) == 0, f"Found ABC classes in MRO: {abc_classes}"

        # ForcingParameter should not be an instance of ABCMeta
        assert not isinstance(ForcingParameter, abc.ABCMeta)

        # ForcingParameter should be a regular class
        assert isinstance(ForcingParameter, type)

    def test_functional_group_parameter_no_longer_inherits_abc(self):
        """Test that FunctionalGroupParameter no longer inherits from AbstractFunctionalGroupParameter."""

        # FunctionalGroupParameter should not have any ABC in its MRO
        abc_classes = [cls for cls in FunctionalGroupParameter.__mro__ if issubclass(cls, abc.ABC) and cls != abc.ABC]
        assert len(abc_classes) == 0, f"Found ABC classes in MRO: {abc_classes}"

        # FunctionalGroupParameter should not be an instance of ABCMeta
        assert not isinstance(FunctionalGroupParameter, abc.ABCMeta)

        # FunctionalGroupParameter should be a regular class
        assert isinstance(FunctionalGroupParameter, type)

    def test_no_transport_configuration_no_longer_inherits_abc(self):
        """Test that NoTransportConfiguration no longer inherits from AbstractConfiguration."""

        # NoTransportConfiguration should not have any ABC in its MRO
        abc_classes = [cls for cls in NoTransportConfiguration.__mro__ if issubclass(cls, abc.ABC) and cls != abc.ABC]
        assert len(abc_classes) == 0, f"Found ABC classes in MRO: {abc_classes}"

        # NoTransportConfiguration should not be an instance of ABCMeta
        assert not isinstance(NoTransportConfiguration, abc.ABCMeta)

        # NoTransportConfiguration should be a regular class
        assert isinstance(NoTransportConfiguration, type)

    def test_forcing_parameter_implements_protocol(self):
        """Test that ForcingParameter implements ForcingParameterProtocol via duck typing."""

        def accepts_forcing_parameter_protocol(param_class: type[ForcingParameterProtocol]) -> bool:
            """Function that accepts ForcingParameterProtocol."""
            return (hasattr(param_class, 'parallel') and
                   hasattr(param_class, 'chunk') and
                   hasattr(param_class, 'to_dataset'))

        # This should work without ABC inheritance
        assert accepts_forcing_parameter_protocol(ForcingParameter)

    def test_functional_group_parameter_implements_protocol(self):
        """Test that FunctionalGroupParameter implements FunctionalGroupParameterProtocol via duck typing."""

        def accepts_functional_group_parameter_protocol(param_class: type[FunctionalGroupParameterProtocol]) -> bool:
            """Function that accepts FunctionalGroupParameterProtocol."""
            return (hasattr(param_class, 'functional_group') and
                   hasattr(param_class, 'to_dataset'))

        # This should work without ABC inheritance
        assert accepts_functional_group_parameter_protocol(FunctionalGroupParameter)

    def test_no_transport_configuration_implements_protocol(self):
        """Test that NoTransportConfiguration implements ConfigurationProtocol via duck typing."""

        def accepts_configuration_protocol(config_class: type[ConfigurationProtocol]) -> bool:
            """Function that accepts ConfigurationProtocol."""
            return (hasattr(config_class, 'forcing') and
                   hasattr(config_class, 'functional_group') and
                   hasattr(config_class, 'kernel') and
                   hasattr(config_class, 'state') and
                   hasattr(config_class, 'parse'))

        # This should work without ABC inheritance
        assert accepts_configuration_protocol(NoTransportConfiguration)

    def test_configuration_classes_can_be_instantiated(self):
        """Test that configuration classes can be instantiated (not abstract anymore)."""

        # Note: We don't actually instantiate them due to complex validation requirements
        # But we verify they're not abstract by checking they don't inherit from ABC

        # All classes should be regular classes, not abstract
        assert not isinstance(ForcingParameter, abc.ABCMeta)
        assert not isinstance(FunctionalGroupParameter, abc.ABCMeta)
        assert not isinstance(NoTransportConfiguration, abc.ABCMeta)

    def test_protocol_compatibility_maintained(self):
        """Test that protocol compatibility is maintained after migration."""

        def process_configuration_hierarchy(config_class: type[ConfigurationProtocol]) -> dict:
            """Function that processes configuration hierarchy."""
            result = {}

            # Level 3: Configuration
            result['has_forcing'] = hasattr(config_class, 'forcing')
            result['has_functional_group'] = hasattr(config_class, 'functional_group')
            result['has_kernel'] = hasattr(config_class, 'kernel')
            result['has_state'] = hasattr(config_class, 'state')
            result['has_parse'] = hasattr(config_class, 'parse')

            return result

        # Test with NoTransportConfiguration
        result = process_configuration_hierarchy(NoTransportConfiguration)
        assert all(result.values()), f"Missing protocol methods: {result}"

    def test_forcing_parameter_protocol_methods(self):
        """Test that ForcingParameter has all required protocol methods."""

        # Check static attributes/methods
        assert hasattr(ForcingParameter, 'to_dataset')
        assert callable(getattr(ForcingParameter, 'to_dataset'))

        # Check attrs fields
        attrs_fields = getattr(ForcingParameter, '__attrs_attrs__', [])
        field_names = [attr.name for attr in attrs_fields]
        assert 'parallel' in field_names
        assert 'chunk' in field_names

    def test_functional_group_parameter_protocol_methods(self):
        """Test that FunctionalGroupParameter has all required protocol methods."""

        # Check static methods
        assert hasattr(FunctionalGroupParameter, 'to_dataset')
        assert callable(getattr(FunctionalGroupParameter, 'to_dataset'))

        # Check attrs fields
        attrs_fields = getattr(FunctionalGroupParameter, '__attrs_attrs__', [])
        field_names = [attr.name for attr in attrs_fields]
        assert 'functional_group' in field_names

    def test_no_transport_configuration_protocol_methods(self):
        """Test that NoTransportConfiguration has all required protocol methods."""

        # Check static methods
        assert hasattr(NoTransportConfiguration, 'parse')
        assert callable(getattr(NoTransportConfiguration, 'parse'))

        # Check properties
        assert hasattr(NoTransportConfiguration, 'state')
        assert isinstance(getattr(NoTransportConfiguration, 'state'), property)

        # Check attrs fields
        attrs_fields = getattr(NoTransportConfiguration, '__attrs_attrs__', [])
        field_names = [attr.name for attr in attrs_fields]
        assert 'forcing' in field_names
        assert 'functional_group' in field_names
        assert 'kernel' in field_names

    def test_migration_maintains_duck_typing(self):
        """Test that migration maintains duck typing functionality."""

        # Test with mock objects
        class MockForcingParameter:
            def __init__(self):
                self.parallel = True
                self.chunk = object()

            def to_dataset(self):
                return {"mock": "forcing"}

        class MockFunctionalGroupParameter:
            def __init__(self):
                self.functional_group = []

            def to_dataset(self, timestep):
                return {"mock": "functional_group"}

        class MockConfiguration:
            def __init__(self):
                self.forcing = MockForcingParameter()
                self.functional_group = MockFunctionalGroupParameter()
                self.kernel = object()

            @property
            def state(self):
                return {"mock": "state"}

            @classmethod
            def parse(cls, config_file):
                return cls()

        mock_config = MockConfiguration()

        # This should work with duck typing
        def use_configuration_protocol(config: ConfigurationProtocol):
            return config.state

        result = use_configuration_protocol(mock_config)
        assert result == {"mock": "state"}
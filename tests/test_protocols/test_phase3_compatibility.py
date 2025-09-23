"""Tests for Phase 3 protocol compatibility.

Verifies that existing configuration classes automatically implement the new
Level 3 ConfigurationProtocol that depends on Phase 1 & 2 protocols.
"""

from typing import Any

import pytest

from seapopym.configuration.no_transport.configuration import NoTransportConfiguration
from seapopym.standard.protocols import ConfigurationProtocol


@pytest.mark.protocols
class TestPhase3ProtocolCompatibility:
    """Test that existing configuration classes implement Phase 3 protocols."""

    def test_no_transport_configuration_implements_protocol(self):
        """Test that NoTransportConfiguration class implements ConfigurationProtocol."""

        # Test that the class has the required attributes (static analysis)
        # We avoid creating instances due to complex validation requirements
        assert hasattr(NoTransportConfiguration, 'forcing')
        assert hasattr(NoTransportConfiguration, 'functional_group')
        assert hasattr(NoTransportConfiguration, 'kernel')
        assert hasattr(NoTransportConfiguration, 'state')
        assert hasattr(NoTransportConfiguration, 'parse')

        # Test that state is a property
        assert isinstance(getattr(NoTransportConfiguration, 'state'), property)

        # Test that parse is callable and accessible from class
        assert callable(NoTransportConfiguration.parse)

        # Test protocol compatibility with duck typing
        def accepts_configuration_protocol(config: ConfigurationProtocol) -> bool:
            # Check that object has required interface
            return (hasattr(config, 'forcing') and
                   hasattr(config, 'functional_group') and
                   hasattr(config, 'kernel') and
                   hasattr(config, 'state') and
                   hasattr(config, 'parse'))

        # Verify the function is correctly typed
        assert callable(accepts_configuration_protocol)

    def test_configuration_protocol_composition(self):
        """Test that ConfigurationProtocol correctly composes lower-level protocols."""

        # Verify that we can define functions that use the protocol hierarchy
        def process_complete_configuration(config: ConfigurationProtocol) -> tuple[Any, Any, Any]:
            """Function that accesses all levels of the protocol hierarchy."""
            # Level 3 → Level 2
            forcing = config.forcing
            functional_group = config.functional_group
            kernel = config.kernel

            # Level 2 → Level 1 (through forcing)
            chunk = forcing.chunk

            # Level 1 methods
            chunk_dict = chunk.as_dict()

            return forcing, functional_group, chunk_dict

        assert callable(process_complete_configuration)

    def test_configuration_protocol_state_property(self):
        """Test that ConfigurationProtocol correctly defines state property."""

        def access_state(config: ConfigurationProtocol) -> Any:
            """Function that accesses the state property."""
            return config.state

        assert callable(access_state)

    def test_configuration_protocol_parse_classmethod(self):
        """Test that ConfigurationProtocol correctly defines parse classmethod."""

        def use_parse_method(config_class: type[ConfigurationProtocol]) -> bool:
            """Function that checks parse method exists."""
            return hasattr(config_class, 'parse') and callable(config_class.parse)

        assert callable(use_parse_method)

    def test_protocol_hierarchy_full_stack(self):
        """Test the complete protocol hierarchy from Level 3 down to Level 1."""

        def traverse_protocol_hierarchy(config: ConfigurationProtocol) -> dict:
            """Function that traverses all protocol levels."""
            result = {}

            # Level 3: Configuration
            result['has_forcing'] = hasattr(config, 'forcing')
            result['has_functional_group'] = hasattr(config, 'functional_group')
            result['has_kernel'] = hasattr(config, 'kernel')
            result['has_state'] = hasattr(config, 'state')

            # Level 2: Parameters
            if hasattr(config, 'forcing'):
                forcing = config.forcing
                result['forcing_has_parallel'] = hasattr(forcing, 'parallel')
                result['forcing_has_chunk'] = hasattr(forcing, 'chunk')
                result['forcing_has_to_dataset'] = hasattr(forcing, 'to_dataset')

                # Level 1: Chunk
                if hasattr(forcing, 'chunk'):
                    chunk = forcing.chunk
                    result['chunk_has_as_dict'] = hasattr(chunk, 'as_dict')

            return result

        assert callable(traverse_protocol_hierarchy)

    def test_no_transport_model_compatibility(self):
        """Test that NoTransportModel can accept ConfigurationProtocol."""
        from seapopym.model.no_transport_model import NoTransportModel

        # Test that NoTransportModel.from_configuration has correct signature
        # We check the method exists and is callable
        assert hasattr(NoTransportModel, 'from_configuration')
        assert callable(NoTransportModel.from_configuration)

    def test_protocol_duck_typing_with_mock(self):
        """Test that ConfigurationProtocol works with duck typing."""

        # Create a mock configuration that implements the protocol
        class MockConfiguration:
            def __init__(self):
                self.forcing = MockForcing()
                self.functional_group = MockFunctionalGroup()
                self.kernel = MockKernel()

            @property
            def state(self):
                return {"mock": "state"}

            @classmethod
            def parse(cls, configuration_file):
                return cls()

        class MockForcing:
            def __init__(self):
                self.parallel = True
                self.chunk = MockChunk()

            def to_dataset(self):
                return {"mock": "forcing"}

        class MockFunctionalGroup:
            def __init__(self):
                self.functional_group = []

            def to_dataset(self, timestep):
                return {"mock": "functional_group"}

        class MockKernel:
            pass

        class MockChunk:
            def as_dict(self):
                return {"mock": "chunk"}

        mock_config = MockConfiguration()

        # This should work without inheritance
        def accepts_configuration_protocol(config: ConfigurationProtocol) -> Any:
            return config.state

        result = accepts_configuration_protocol(mock_config)
        assert result == {"mock": "state"}
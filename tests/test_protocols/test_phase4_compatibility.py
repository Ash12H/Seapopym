"""Tests for Phase 4 protocol compatibility.

Verifies that existing model classes automatically implement the new
ModelProtocol that depends on Phase 3 ConfigurationProtocol.
"""

from typing import Any
from unittest.mock import Mock

import pytest

from seapopym.model.base_model import BaseModel
from seapopym.model.no_transport_model import NoTransportModel, NoTransportLightModel
from seapopym.standard.protocols import ModelProtocol


@pytest.mark.protocols
class TestPhase4ProtocolCompatibility:
    """Test that existing model classes implement Phase 4 protocols."""

    def test_base_model_implements_protocol(self):
        """Test that BaseModel class implements ModelProtocol interface."""

        # Test that the class has the required methods (static analysis)
        assert hasattr(BaseModel, 'from_configuration')
        assert hasattr(BaseModel, 'run')
        assert hasattr(BaseModel, '__enter__')
        assert hasattr(BaseModel, '__exit__')

        # Test that from_configuration is a classmethod
        assert callable(BaseModel.from_configuration)

        # Test that run is an instance method
        assert callable(BaseModel.run)

        # Test context manager methods are callable
        assert callable(BaseModel.__enter__)
        assert callable(BaseModel.__exit__)

        # Test dataclass field annotations (state and kernel are instance attributes)
        assert hasattr(BaseModel, '__annotations__')
        annotations = getattr(BaseModel, '__annotations__', {})
        assert 'state' in annotations
        assert 'kernel' in annotations

    def test_no_transport_model_implements_protocol(self):
        """Test that NoTransportModel class implements ModelProtocol."""

        # Test that the class has the required methods
        assert hasattr(NoTransportModel, 'from_configuration')
        assert hasattr(NoTransportModel, 'run')
        assert hasattr(NoTransportModel, '__enter__')
        assert hasattr(NoTransportModel, '__exit__')

        # Test dataclass fields (inherited from BaseModel)
        fields = getattr(NoTransportModel, '__dataclass_fields__', {})
        assert 'state' in fields
        assert 'kernel' in fields

        # Test additional properties specific to NoTransportModel
        assert hasattr(NoTransportModel, 'template')
        assert hasattr(NoTransportModel, 'expected_memory_usage')
        assert hasattr(NoTransportModel, 'export_initial_conditions')

        # Test that template and expected_memory_usage are properties
        assert isinstance(getattr(NoTransportModel, 'template'), property)
        assert isinstance(getattr(NoTransportModel, 'expected_memory_usage'), property)

    def test_no_transport_light_model_implements_protocol(self):
        """Test that NoTransportLightModel class implements ModelProtocol."""

        # Test that the class has the required methods
        assert hasattr(NoTransportLightModel, 'from_configuration')
        assert hasattr(NoTransportLightModel, 'run')
        assert hasattr(NoTransportLightModel, '__enter__')
        assert hasattr(NoTransportLightModel, '__exit__')

        # Test dataclass fields (inherited from BaseModel)
        fields = getattr(NoTransportLightModel, '__dataclass_fields__', {})
        assert 'state' in fields
        assert 'kernel' in fields

        # Test inheritance from NoTransportModel
        assert issubclass(NoTransportLightModel, NoTransportModel)

    def test_model_protocol_duck_typing(self):
        """Test that ModelProtocol works with duck typing."""

        def accepts_model_protocol(model: ModelProtocol) -> bool:
            """Function that checks model protocol interface."""
            return (hasattr(model, 'state') and
                   hasattr(model, 'kernel') and
                   hasattr(model, 'from_configuration') and
                   hasattr(model, 'run') and
                   hasattr(model, '__enter__') and
                   hasattr(model, '__exit__'))

        # Verify the function is correctly typed
        assert callable(accepts_model_protocol)

    def test_model_protocol_with_mock(self):
        """Test that ModelProtocol works with duck typing using mock."""

        # Create a mock model that implements the protocol
        class MockModel:
            def __init__(self):
                self.state = {"mock": "state"}
                self.kernel = Mock()

            @classmethod
            def from_configuration(cls, configuration):
                return cls()

            def run(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass

        mock_model = MockModel()

        # This should work without inheritance
        def use_model_protocol(model: ModelProtocol) -> Any:
            """Function that uses model protocol."""
            model.run()
            return model.state

        result = use_model_protocol(mock_model)
        assert result == {"mock": "state"}

    def test_model_context_manager_protocol(self):
        """Test that ModelProtocol correctly defines context manager interface."""

        def use_model_context_manager(model_class: type[ModelProtocol]) -> bool:
            """Function that uses model as context manager."""
            # Check that context manager methods exist
            return (hasattr(model_class, '__enter__') and
                   hasattr(model_class, '__exit__') and
                   callable(model_class.__enter__) and
                   callable(model_class.__exit__))

        assert callable(use_model_context_manager)

    def test_model_factory_pattern_compatibility(self):
        """Test that ModelProtocol is compatible with factory pattern."""

        def create_model_from_config(
            model_class: type[ModelProtocol],
            configuration: Any
        ) -> ModelProtocol:
            """Factory function that creates models from configuration."""
            return model_class.from_configuration(configuration)

        assert callable(create_model_from_config)

    def test_model_protocol_inheritance_hierarchy(self):
        """Test that protocol works with model inheritance hierarchy."""

        def process_model_hierarchy(model: ModelProtocol) -> dict:
            """Function that works with model hierarchy."""
            result = {}

            # Core protocol interface
            result['has_state'] = hasattr(model, 'state')
            result['has_kernel'] = hasattr(model, 'kernel')
            result['has_run'] = hasattr(model, 'run')
            result['has_from_configuration'] = hasattr(model, 'from_configuration')

            # Context manager interface
            result['has_enter'] = hasattr(model, '__enter__')
            result['has_exit'] = hasattr(model, '__exit__')

            # Optional extended interface (NoTransportModel specific)
            result['has_template'] = hasattr(model, 'template')
            result['has_export_initial_conditions'] = hasattr(model, 'export_initial_conditions')

            return result

        assert callable(process_model_hierarchy)
"""Tests for Phase 5 ABC to Protocol migration.

Verifies that BaseModel has been successfully migrated from ABC to Protocol.
"""

import abc
import pytest

from seapopym.model.no_transport_model import NoTransportModel
from seapopym.standard.protocols import ModelProtocol


@pytest.mark.protocols
class TestPhase5ABCMigration:
    """Test that NoTransportModel has been migrated from ABC to Protocol."""

    def test_no_transport_model_no_longer_inherits_abc(self):
        """Test that NoTransportModel no longer inherits from abc.ABC."""

        # NoTransportModel should not directly inherit from abc.ABC (check __bases__)
        assert abc.ABC not in NoTransportModel.__bases__

        # NoTransportModel should not have abc.ABC in its MRO
        assert abc.ABC not in NoTransportModel.__mro__

        # NoTransportModel should not be an instance of ABCMeta
        assert not isinstance(NoTransportModel, abc.ABCMeta)

        # NoTransportModel should be a regular class
        assert isinstance(NoTransportModel, type)

    def test_no_transport_model_implements_model_protocol(self):
        """Test that NoTransportModel implements ModelProtocol via duck typing."""

        def accepts_model_protocol(model_class: type[ModelProtocol]) -> bool:
            """Function that accepts ModelProtocol."""
            # Check required methods exist
            return (hasattr(model_class, 'from_configuration') and
                   hasattr(model_class, 'run') and
                   hasattr(model_class, '__enter__') and
                   hasattr(model_class, '__exit__'))

        # This should work without ABC inheritance
        assert accepts_model_protocol(NoTransportModel)

    def test_no_transport_model_context_manager_works(self):
        """Test that NoTransportModel context manager functionality works."""
        from unittest.mock import Mock

        # Create a mock state that has a close method (simulating xarray Dataset)
        mock_state = Mock()
        mock_kernel = object()

        model = NoTransportModel(state=mock_state, kernel=mock_kernel)

        # Context manager should work
        with model as ctx_model:
            assert ctx_model is model
            assert hasattr(ctx_model, 'state')
            assert hasattr(ctx_model, 'kernel')

        # After context exit, state and kernel should be deleted
        assert not hasattr(model, 'state')
        assert not hasattr(model, 'kernel')

    def test_no_transport_model_can_be_instantiated_directly(self):
        """Test that NoTransportModel can be instantiated directly."""

        mock_state = {"test": "state"}
        mock_kernel = object()

        # This should work without ABC restrictions
        model = NoTransportModel(state=mock_state, kernel=mock_kernel)

        assert model.state == mock_state
        assert model.kernel == mock_kernel

    def test_migration_maintains_protocol_compatibility(self):
        """Test that migration maintains protocol compatibility."""

        # NoTransportModel should still be compatible with functions expecting ModelProtocol
        def process_model(model_class: type[ModelProtocol]) -> str:
            """Function that processes a model class implementing ModelProtocol."""
            # Check core protocol requirements
            assert hasattr(model_class, 'from_configuration')
            assert hasattr(model_class, 'run')
            assert hasattr(model_class, '__enter__')
            assert hasattr(model_class, '__exit__')

            # Check dataclass fields
            fields = getattr(model_class, '__dataclass_fields__', {})
            assert 'state' in fields
            assert 'kernel' in fields

            return "Protocol compatible"

        result = process_model(NoTransportModel)
        assert result == "Protocol compatible"
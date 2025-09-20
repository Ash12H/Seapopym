"""Tests for Phase 5 ABC to Protocol migration.

Verifies that BaseModel has been successfully migrated from ABC to Protocol.
"""

import abc
import pytest

from seapopym.model.base_model import BaseModel
from seapopym.standard.protocols import ModelProtocol


@pytest.mark.protocols
class TestPhase5ABCMigration:
    """Test that BaseModel has been migrated from ABC to Protocol."""

    def test_base_model_no_longer_inherits_abc(self):
        """Test that BaseModel no longer inherits from abc.ABC."""

        # BaseModel should not directly inherit from abc.ABC (check __bases__)
        assert abc.ABC not in BaseModel.__bases__

        # BaseModel should not have abc.ABC in its MRO
        assert abc.ABC not in BaseModel.__mro__

        # BaseModel should not be an instance of ABCMeta
        assert not isinstance(BaseModel, abc.ABCMeta)

        # BaseModel should be a regular class
        assert isinstance(BaseModel, type)

        # BaseModel should be instantiable (not abstract)
        mock_state = {"test": "state"}
        mock_kernel = object()
        model = BaseModel(state=mock_state, kernel=mock_kernel)
        assert model is not None

    def test_base_model_implements_model_protocol(self):
        """Test that BaseModel implements ModelProtocol via duck typing."""

        def accepts_model_protocol(model_class: type[ModelProtocol]) -> bool:
            """Function that accepts ModelProtocol."""
            # Check required methods exist
            return (hasattr(model_class, 'from_configuration') and
                   hasattr(model_class, 'run') and
                   hasattr(model_class, '__enter__') and
                   hasattr(model_class, '__exit__'))

        # This should work without ABC inheritance
        assert accepts_model_protocol(BaseModel)

    def test_base_model_abstract_methods_raise_not_implemented(self):
        """Test that BaseModel's abstract methods raise NotImplementedError."""

        # Creating BaseModel directly should be possible (no ABC preventing it)
        mock_state = {"test": "state"}
        mock_kernel = object()

        model = BaseModel(state=mock_state, kernel=mock_kernel)

        # from_configuration should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="must implement from_configuration"):
            BaseModel.from_configuration(None)

        # run should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="must implement run"):
            model.run()

    def test_base_model_context_manager_still_works(self):
        """Test that BaseModel context manager functionality still works."""

        mock_state = {"test": "state"}
        mock_kernel = object()

        model = BaseModel(state=mock_state, kernel=mock_kernel)

        # Context manager should work
        with model as ctx_model:
            assert ctx_model is model
            assert hasattr(ctx_model, 'state')
            assert hasattr(ctx_model, 'kernel')

        # After context exit, state and kernel should be deleted
        assert not hasattr(model, 'state')
        assert not hasattr(model, 'kernel')

    def test_base_model_can_be_instantiated_directly(self):
        """Test that BaseModel can be instantiated directly (not abstract anymore)."""

        mock_state = {"test": "state"}
        mock_kernel = object()

        # This should work without ABC restrictions
        model = BaseModel(state=mock_state, kernel=mock_kernel)

        assert model.state == mock_state
        assert model.kernel == mock_kernel

    def test_migration_maintains_protocol_compatibility(self):
        """Test that migration maintains protocol compatibility."""

        # BaseModel should still be compatible with functions expecting ModelProtocol
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

        result = process_model(BaseModel)
        assert result == "Protocol compatible"
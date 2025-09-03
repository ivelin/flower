"""Unit tests for error handling scenarios in SmolVLA."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from smolvla_example.client_app import SmolVLAClient


@pytest.mark.unit
class TestModelLoadingErrors:
    """Test error handling during model loading."""

    @pytest.fixture
    def client_config(self):
        """Default client configuration for tests."""
        return {
            "model_name": "lerobot/smolvla_base",
            "device": "cpu",
            "partition_id": 0,
            "num_partitions": 2
        }

    def test_model_loading_import_error(self, client_config):
        """Test handling of import errors during model loading."""
        with patch.dict('sys.modules', {'transformers': None}):
            with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model:
                mock_model.from_pretrained.side_effect = ImportError("transformers not available")

                # Should handle gracefully
                client = SmolVLAClient(**client_config)

                assert client.model is None
                assert client.processor is None

    def test_model_loading_network_error(self, client_config):
        """Test handling of network errors during model loading."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model:
            mock_model.from_pretrained.side_effect = ConnectionError("Network timeout")

            client = SmolVLAClient(**client_config)

            assert client.model is None

    def test_model_loading_invalid_model_name(self, client_config):
        """Test handling of invalid model names."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model:
            mock_model.from_pretrained.side_effect = ValueError("Invalid model name")

            client = SmolVLAClient(**client_config)

            assert client.model is None

    def test_processor_loading_failure(self, client_config):
        """Test handling of processor loading failures."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model, \
             patch('smolvla_example.client_app.AutoProcessor') as mock_processor:

            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_processor.from_pretrained.side_effect = Exception("Processor loading failed")

            client = SmolVLAClient(**client_config)

            # Model should still be loaded, but processor should be None
            assert client.model is not None
            assert client.processor is None


@pytest.mark.unit
class TestDatasetLoadingErrors:
    """Test error handling during dataset loading."""

    @pytest.fixture
    def client_config(self):
        """Default client configuration for tests."""
        return {
            "model_name": "lerobot/smolvla_base",
            "device": "cpu",
            "partition_id": 0,
            "num_partitions": 2
        }

    def test_dataset_import_error(self, client_config):
        """Test handling of dataset import errors."""
        with patch.dict('sys.modules', {
            'lerobot_dataset_partitioner': None,
            'federated_lerobot_dataset': None
        }):
            with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model, \
                 patch('smolvla_example.client_app.AutoProcessor'), \
                 patch('smolvla_example.client_app.LeRobotDatasetPartitioner') as mock_partitioner, \
                 patch('smolvla_example.client_app.FederatedLeRobotDataset') as mock_federated:

                mock_model_instance = Mock()
                mock_model.from_pretrained.return_value = mock_model_instance

                # Make the mocked classes raise ImportError
                mock_partitioner.side_effect = ImportError("Dataset partitioner not available")
                mock_federated.side_effect = ImportError("Federated dataset not available")

                client = SmolVLAClient(**client_config)

                # Should handle import errors gracefully
                assert client.train_loader is None

    def test_dataset_loading_connection_error(self, client_config):
        """Test handling of connection errors during dataset loading."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model, \
             patch('smolvla_example.client_app.AutoProcessor'), \
             patch('smolvla_example.client_app.FederatedLeRobotDataset') as mock_federated:

            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance

            mock_federated.side_effect = ConnectionError("Dataset download failed")

            client = SmolVLAClient(**client_config)

            assert client.federated_dataset is None
            assert client.train_loader is None

    def test_dataset_loading_invalid_partition(self, client_config):
        """Test handling of invalid partition IDs."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model, \
             patch('smolvla_example.client_app.AutoProcessor'), \
             patch('smolvla_example.client_app.FederatedLeRobotDataset') as mock_federated_class:

            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance

            mock_dataset = Mock()
            mock_dataset.load_partition.side_effect = IndexError("Invalid partition ID")
            mock_federated_class.return_value = mock_dataset

            client = SmolVLAClient(**client_config)

            assert client.train_loader is None

    def test_dataloader_creation_failure(self, client_config):
        """Test handling of DataLoader creation failures."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model, \
             patch('smolvla_example.client_app.AutoProcessor'), \
             patch('smolvla_example.client_app.FederatedLeRobotDataset') as mock_federated_class, \
             patch('smolvla_example.client_app.DataLoader') as mock_dataloader:

            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance

            mock_dataset = Mock()
            mock_dataset.load_partition.return_value = [Mock()]
            mock_federated_class.return_value = mock_dataset

            mock_dataloader.side_effect = RuntimeError("DataLoader creation failed")

            client = SmolVLAClient(**client_config)

            assert client.train_loader is None


@pytest.mark.unit
class TestTrainingErrors:
    """Test error handling during training."""

    @pytest.fixture
    def client_config(self):
        """Default client configuration for tests."""
        return {
            "model_name": "lerobot/smolvla_base",
            "device": "cpu",
            "partition_id": 0,
            "num_partitions": 2
        }

    @pytest.fixture
    def mock_client(self, client_config):
        """Create a mock client for training tests."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model, \
              patch('smolvla_example.client_app.AutoProcessor'), \
              patch('smolvla_example.client_app.LeRobotDatasetPartitioner'), \
              patch('smolvla_example.client_app.FederatedLeRobotDataset'):

            mock_model_instance = Mock()
            mock_model_instance.vision_encoder = Mock()
            mock_model_instance.vision_encoder.parameters = Mock()

            # Mock parameters for optimizer
            mock_param = Mock()
            mock_param.requires_grad = True
            mock_model_instance.parameters.return_value = [mock_param]

            # Mock optimizer
            mock_optimizer = Mock()
            mock_model_instance.optimizer = mock_optimizer

            mock_model.from_pretrained.return_value = mock_model_instance

            mock_dataset = Mock()
            mock_dataset.load_partition.return_value = [Mock()]

            client = SmolVLAClient(**client_config)
            return client

    def test_fit_with_invalid_parameters(self, mock_client):
        """Test fit with invalid parameters."""
        from flwr.common import FitIns, Parameters

        # Create invalid parameters (empty tensors)
        invalid_parameters = Parameters([], "numpy")
        fit_ins = FitIns(parameters=invalid_parameters, config={})

        result = mock_client.fit(fit_ins)

        # Should handle gracefully
        assert result.status.code.value == 0  # OK
        assert result.num_examples >= 0

    def test_fit_with_model_forward_error(self, mock_client):
        """Test fit with model forward pass errors."""
        from flwr.common import FitIns, Parameters

        # Mock parameters
        mock_parameters = Parameters([np.array([1, 2, 3])], "numpy")
        fit_ins = FitIns(parameters=mock_parameters, config={"local_epochs": 1})

        # Mock model to raise error during forward pass
        mock_client.model.side_effect = RuntimeError("Model forward pass failed")

        result = mock_client.fit(fit_ins)

        # Should handle gracefully
        assert result.status.code.value == 0  # OK
        assert "error" in result.metrics

    def test_fit_with_optimizer_error(self, mock_client):
        """Test fit with optimizer errors."""
        from flwr.common import FitIns, Parameters

        # Mock parameters
        mock_parameters = Parameters([np.array([1, 2, 3])], "numpy")
        fit_ins = FitIns(parameters=mock_parameters, config={"local_epochs": 1})

        # Mock optimizer to raise error
        mock_client.optimizer.step.side_effect = RuntimeError("Optimizer step failed")

        result = mock_client.fit(fit_ins)

        # Should handle gracefully
        assert result.status.code.value == 0  # OK
        assert "error" in result.metrics

    def test_evaluate_with_invalid_parameters(self, mock_client):
        """Test evaluate with invalid parameters."""
        from flwr.common import EvaluateIns, Parameters

        # Create invalid parameters
        invalid_parameters = Parameters([], "numpy")
        evaluate_ins = EvaluateIns(parameters=invalid_parameters, config={})

        result = mock_client.evaluate(evaluate_ins)

        # Should handle gracefully
        assert result.status.code.value == 0  # OK
        assert result.num_examples >= 0

    def test_evaluate_with_model_error(self, mock_client):
        """Test evaluate with model errors."""
        from flwr.common import EvaluateIns, Parameters

        # Mock parameters
        mock_parameters = Parameters([np.array([1, 2, 3])], "numpy")
        evaluate_ins = EvaluateIns(parameters=mock_parameters, config={})

        # Mock model to raise error during evaluation
        mock_client.model.side_effect = RuntimeError("Model evaluation failed")

        result = mock_client.evaluate(evaluate_ins)

        # Should handle gracefully
        assert result.status.code.value == 0  # OK
        assert "error" in result.metrics


@pytest.mark.unit
class TestCheckpointErrors:
    """Test error handling during checkpoint operations."""

    @pytest.fixture
    def client_config(self):
        """Default client configuration for tests."""
        return {
            "model_name": "lerobot/smolvla_base",
            "device": "cpu",
            "partition_id": 0,
            "num_partitions": 2
        }

    @pytest.fixture
    def mock_client(self, client_config):
        """Create a mock client for checkpoint tests."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model, \
             patch('smolvla_example.client_app.AutoProcessor'), \
             patch('smolvla_example.client_app.LeRobotDatasetPartitioner'), \
             patch('smolvla_example.client_app.FederatedLeRobotDataset'):

            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance

            mock_dataset = Mock()
            mock_dataset.load_partition.return_value = [Mock()]

            client = SmolVLAClient(**client_config)
            return client

    def test_save_checkpoint_io_error(self, mock_client):
        """Test handling of IO errors during checkpoint saving."""
        with patch('torch.save') as mock_save:
            mock_save.side_effect = IOError("Disk write failed")

            # Should not raise exception
            mock_client._save_checkpoint("test_checkpoint")

    def test_save_checkpoint_without_model(self, client_config):
        """Test checkpoint saving when model is not available."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model:
            mock_model.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**client_config)

            # Should not raise exception
            client._save_checkpoint("test_checkpoint")

    def test_load_checkpoint_file_not_found(self, mock_client):
        """Test handling of missing checkpoint files."""
        from pathlib import Path

        checkpoint_path = Path("nonexistent_checkpoint.pt")

        # Should not raise exception
        mock_client._load_checkpoint(checkpoint_path)

    def test_load_checkpoint_corrupted_file(self, mock_client):
        """Test handling of corrupted checkpoint files."""
        from pathlib import Path

        with patch('torch.load') as mock_load:
            mock_load.side_effect = Exception("Corrupted checkpoint file")

            checkpoint_path = Path("corrupted_checkpoint.pt")

            # Should not raise exception
            mock_client._load_checkpoint(checkpoint_path)
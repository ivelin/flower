"""Unit tests for SmolVLAClient class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np

from smolvla_example.client_app import SmolVLAClient, get_device


class TestGetDevice:
    """Test cases for get_device function."""

    @patch('torch.cuda.is_available', return_value=True)
    def test_get_device_auto_with_cuda(self, mock_cuda_available):
        """Test get_device with auto when CUDA is available."""
        result = get_device("auto")
        assert result == "cuda"

    @patch('torch.cuda.is_available', return_value=False)
    def test_get_device_auto_without_cuda(self, mock_cuda_available):
        """Test get_device with auto when CUDA is not available."""
        result = get_device("auto")
        assert result == "cpu"

    def test_get_device_cpu(self):
        """Test get_device with explicit cpu."""
        result = get_device("cpu")
        assert result == "cpu"

    def test_get_device_cuda(self):
        """Test get_device with explicit cuda."""
        result = get_device("cuda")
        assert result == "cuda"


class TestSmolVLAClient:
    """Test cases for SmolVLAClient class."""

    @pytest.fixture
    def client_config(self):
        """Default client configuration for tests."""
        return {
            "model_name": "lerobot/smolvla_base",
            "device": "cpu",
            "partition_id": 0,
            "num_partitions": 2
        }

    @patch('smolvla_example.client_app.AutoModelForVision2Seq')
    @patch('smolvla_example.client_app.AutoProcessor')
    @patch('smolvla_example.client_app.LeRobotDatasetPartitioner')
    @patch('smolvla_example.client_app.FederatedLeRobotDataset')
    def test_client_initialization_success(self, mock_federated_dataset, mock_partitioner,
                                         mock_processor, mock_model, client_config):
        """Test successful client initialization."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_dataset = Mock()
        mock_federated_dataset.return_value = mock_dataset
        mock_dataset.load_partition.return_value = [Mock()]

        # Create client
        client = SmolVLAClient(**client_config)

        # Assertions
        assert client.model_name == client_config["model_name"]
        assert client.device == client_config["device"]
        assert client.partition_id == client_config["partition_id"]
        assert client.num_partitions == client_config["num_partitions"]
        assert client.model is not None
        assert client.processor is not None
        assert client.federated_dataset is not None

    @patch('smolvla_example.client_app.AutoModelForVision2Seq')
    @patch('smolvla_example.client_app.AutoProcessor')
    def test_client_initialization_model_failure(self, mock_processor, mock_model, client_config):
        """Test client initialization when model loading fails."""
        # Setup mock to raise exception
        mock_model.from_pretrained.side_effect = Exception("Model loading failed")

        # Create client - should handle exception gracefully
        client = SmolVLAClient(**client_config)

        # Assertions
        assert client.model is None
        assert client.processor is None

    @patch('smolvla_example.client_app.AutoModelForVision2Seq')
    @patch('smolvla_example.client_app.AutoProcessor')
    @patch('smolvla_example.client_app.LeRobotDatasetPartitioner')
    @patch('smolvla_example.client_app.FederatedLeRobotDataset')
    def test_client_initialization_dataset_failure(self, mock_federated_dataset, mock_partitioner,
                                                 mock_processor, mock_model, client_config):
        """Test client initialization when dataset loading fails."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance

        # Make dataset loading fail
        mock_federated_dataset.side_effect = Exception("Dataset loading failed")

        # Create client - should handle exception gracefully
        client = SmolVLAClient(**client_config)

        # Assertions
        assert client.train_loader is None
        assert client.federated_dataset is None

    def test_get_parameters_with_model(self, client_config):
        """Test get_parameters when model is available."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('smolvla_example.client_app.AutoProcessor'), \
             patch('smolvla_example.client_app.LeRobotDatasetPartitioner'), \
             patch('smolvla_example.client_app.FederatedLeRobotDataset'):

            mock_model = Mock()
            mock_model.state_dict.return_value = {'param1': Mock(), 'param2': Mock()}
            mock_model_class.from_pretrained.return_value = mock_model

            client = SmolVLAClient(**client_config)

            # Mock numpy conversion
            with patch('numpy.array') as mock_numpy:
                mock_numpy.return_value = np.array([1, 2, 3])

                from flwr.common import GetParametersIns
                result = client.get_parameters(GetParametersIns(config={}))

                assert result.status.code.value == 0  # OK
                assert result.parameters is not None

    def test_get_parameters_without_model(self, client_config):
        """Test get_parameters when model is not available."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**client_config)

            from flwr.common import GetParametersIns
            result = client.get_parameters(GetParametersIns(config={}))

            assert result.status.code.value == 0  # OK
            assert len(result.parameters.tensors) == 0

    def test_set_parameters_with_model(self, client_config):
        """Test set_parameters when model is available."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('smolvla_example.client_app.AutoProcessor'), \
             patch('smolvla_example.client_app.LeRobotDatasetPartitioner'), \
             patch('smolvla_example.client_app.FederatedLeRobotDataset'):

            mock_model = Mock()
            mock_model.load_state_dict.return_value = None
            mock_model_class.from_pretrained.return_value = mock_model

            client = SmolVLAClient(**client_config)

            parameters = [np.array([1, 2, 3]), np.array([4, 5, 6])]

            # Should not raise exception
            client.set_parameters(parameters)

            # Verify load_state_dict was called
            assert mock_model.load_state_dict.called

    def test_set_parameters_without_model(self, client_config):
        """Test set_parameters when model is not available."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**client_config)

            parameters = [np.array([1, 2, 3])]

            # Should not raise exception
            client.set_parameters(parameters)

    @patch('time.time')
    def test_save_checkpoint(self, mock_time, client_config, tmp_path):
        """Test checkpoint saving functionality."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class, \
             patch('smolvla_example.client_app.AutoProcessor'), \
             patch('smolvla_example.client_app.LeRobotDatasetPartitioner'), \
             patch('smolvla_example.client_app.FederatedLeRobotDataset'):

            mock_model = Mock()
            mock_optimizer = Mock()
            mock_model.state_dict.return_value = {'param': Mock()}
            mock_optimizer.state_dict.return_value = {'lr': 0.01}
            mock_model_class.from_pretrained.return_value = mock_model

            client = SmolVLAClient(**client_config)
            client.optimizer = mock_optimizer

            # Mock torch.save
            with patch('torch.save') as mock_save:
                mock_time.return_value = 1234567890.0

                client._save_checkpoint("test_checkpoint")

                # Verify save was called
                assert mock_save.called

    def test_simulate_training_step(self, client_config):
        """Test training step simulation."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**client_config)

            loss = client._simulate_training_step()

            assert isinstance(loss, float)
            assert 0.1 <= loss <= 0.6  # Based on the implementation

    def test_simulate_validation_step(self, client_config):
        """Test validation step simulation."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model_class:
            mock_model_class.from_pretrained.side_effect = Exception("Model failed")

            client = SmolVLAClient(**client_config)

            loss, correct = client._simulate_validation_step()

            assert isinstance(loss, float)
            assert isinstance(correct, int)
            assert 0.1 <= loss <= 0.4  # Based on the implementation
            assert 0 <= correct <= 4  # Based on the implementation
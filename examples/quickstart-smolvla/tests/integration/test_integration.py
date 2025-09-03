"""Integration tests for SmolVLA federated learning workflow."""

import pytest
from unittest.mock import patch, Mock
import numpy as np

from smolvla_example.client_app import SmolVLAClient, get_device


@pytest.mark.integration
class TestDeviceDetection:
    """Test device detection functionality."""

    @patch('smolvla_example.client_app.torch.cuda.is_available', return_value=True)
    def test_device_detection_auto_with_cuda(self, mock_cuda_available):
        """Test device detection with auto when CUDA is available."""
        device = get_device("auto")
        assert device == "cuda"

    @patch('torch.cuda.is_available', return_value=False)
    def test_device_detection_auto_without_cuda(self, mock_cuda_available):
        """Test device detection with auto when CUDA is not available."""
        device = get_device("auto")
        assert device == "cpu"

    def test_device_detection_cpu(self):
        """Test device detection with explicit CPU."""
        device = get_device("cpu")
        assert device == "cpu"

    def test_device_detection_cuda(self):
        """Test device detection with explicit CUDA."""
        device = get_device("cuda")
        assert device == "cuda"


@pytest.mark.integration
class TestSmolVLAClientIntegration:
    """Integration tests for SmolVLAClient."""


    @patch('smolvla_example.client_app.AutoModelForVision2Seq')
    @patch('smolvla_example.client_app.AutoProcessor')
    @patch('smolvla_example.client_app.LeRobotDatasetPartitioner')
    @patch('smolvla_example.client_app.FederatedLeRobotDataset')
    def test_client_initialization_integration(self, mock_federated_dataset, mock_partitioner,
                                             mock_processor, mock_model, client_config):
        """Test full client initialization integration."""
        # Setup mocks for successful initialization
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model_instance.vision_encoder = Mock()
        mock_model_instance.vision_encoder.parameters = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_dataset = Mock()
        mock_federated_dataset.return_value = mock_dataset
        mock_dataset.load_partition.return_value = [Mock()]

        # Test initialization
        client = SmolVLAClient(**client_config)

        assert client.model is not None
        assert client.processor is not None
        assert client.federated_dataset is not None
        assert client.train_loader is not None

    @patch('smolvla_example.client_app.AutoModelForVision2Seq')
    @patch('smolvla_example.client_app.AutoProcessor')
    @patch('smolvla_example.client_app.LeRobotDatasetPartitioner')
    @patch('smolvla_example.client_app.FederatedLeRobotDataset')
    def test_dataset_loading_integration(self, mock_federated_dataset, mock_partitioner,
                                        mock_processor, mock_model, client_config):
        """Test dataset loading integration."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_dataset = Mock()
        mock_federated_dataset.return_value = mock_dataset
        mock_partition = [Mock(), Mock(), Mock()]  # Mock dataset with 3 samples
        mock_dataset.load_partition.return_value = mock_partition

        # Test dataset loading
        client = SmolVLAClient(**client_config)

        assert client.train_loader is not None
        # Verify that the dataset partition was loaded
        mock_dataset.load_partition.assert_called_once_with(client_config["partition_id"])

    @patch('smolvla_example.client_app.AutoModelForVision2Seq')
    @patch('smolvla_example.client_app.AutoProcessor')
    @patch('smolvla_example.client_app.LeRobotDatasetPartitioner')
    @patch('smolvla_example.client_app.FederatedLeRobotDataset')
    def test_model_loading_integration(self, mock_federated_dataset, mock_partitioner,
                                      mock_processor, mock_model, client_config):
        """Test model loading integration."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance

        mock_dataset = Mock()
        mock_federated_dataset.return_value = mock_dataset
        mock_dataset.load_partition.return_value = [Mock()]

        # Test model loading
        client = SmolVLAClient(**client_config)

        # Verify model was loaded with correct parameters
        mock_model.from_pretrained.assert_called_once()
        call_args = mock_model.from_pretrained.call_args
        assert call_args[0][0] == client_config["model_name"]  # First positional arg is model_name
        assert call_args[1]["trust_remote_code"] == True  # trust_remote_code kwarg

        mock_processor.from_pretrained.assert_called_once_with(
            client_config["model_name"],
            trust_remote_code=True
        )

        # Verify vision encoder was frozen
        assert mock_model_instance.vision_encoder.parameters.called

    @patch('smolvla_example.client_app.AutoModelForVision2Seq')
    @patch('smolvla_example.client_app.AutoProcessor')
    @patch('smolvla_example.client_app.LeRobotDatasetPartitioner')
    @patch('smolvla_example.client_app.FederatedLeRobotDataset')
    def test_optimizer_initialization_integration(self, mock_federated_dataset, mock_partitioner,
                                                 mock_processor, mock_model, client_config):
        """Test optimizer initialization integration."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance

        # Mock parameters that require gradients
        mock_param1 = Mock()
        mock_param1.requires_grad = True
        mock_param2 = Mock()
        mock_param2.requires_grad = False
        mock_model_instance.parameters.return_value = [mock_param1, mock_param2]
        mock_model_instance.vision_encoder = Mock()
        mock_model_instance.vision_encoder.parameters = Mock()

        mock_dataset = Mock()
        mock_federated_dataset.return_value = mock_dataset
        mock_dataset.load_partition.return_value = [Mock()]

        # Test optimizer initialization
        client = SmolVLAClient(**client_config)

        assert client.optimizer is not None
        # Verify optimizer was created with only trainable parameters
        # This would require more detailed mocking of torch.optim.Adam

    def test_client_initialization_failure_handling(self, client_config):
        """Test client initialization failure handling."""
        # Test with model loading failure
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model:
            mock_model.from_pretrained.side_effect = Exception("Model loading failed")

            # Should not raise exception
            client = SmolVLAClient(**client_config)

            assert client.model is None
            assert client.processor is None

    def test_dataset_loading_failure_handling(self, client_config):
        """Test dataset loading failure handling."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model, \
             patch('smolvla_example.client_app.AutoProcessor'), \
             patch('smolvla_example.client_app.FederatedLeRobotDataset') as mock_federated:

            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance

            mock_federated.side_effect = Exception("Dataset loading failed")

            # Should not raise exception
            client = SmolVLAClient(**client_config)

            assert client.train_loader is None
            assert client.federated_dataset is None


@pytest.mark.integration
class TestFederatedLearningWorkflow:
    """Test federated learning workflow integration."""

    @pytest.fixture
    def mock_client(self, client_config):
        """Create a mock client for workflow testing."""
        with patch('smolvla_example.client_app.AutoModelForVision2Seq') as mock_model, \
              patch('smolvla_example.client_app.AutoProcessor'), \
              patch('smolvla_example.client_app.LeRobotDatasetPartitioner'), \
              patch('smolvla_example.client_app.FederatedLeRobotDataset'):

            mock_model_instance = Mock()
            mock_model_instance.vision_encoder = Mock()
            mock_model_instance.vision_encoder.parameters = Mock()

            # Mock state_dict to return iterable dict
            mock_tensor = Mock()
            mock_tensor.cpu.return_value.numpy.return_value = [1, 2, 3]
            mock_model_instance.state_dict.return_value = {'param1': mock_tensor, 'param2': mock_tensor}

            # Mock parameters for optimizer
            mock_param = Mock()
            mock_param.requires_grad = True
            mock_model_instance.parameters.return_value = [mock_param]

            mock_model.from_pretrained.return_value = mock_model_instance

            mock_dataset = Mock()
            mock_dataset.load_partition.return_value = [Mock()]

            client = SmolVLAClient(**client_config)
            return client

    def test_get_parameters_workflow(self, mock_client):
        """Test get_parameters in federated workflow."""
        from flwr.common import GetParametersIns

        # Mock the model state dict
        mock_client.model.state_dict.return_value = {
            'param1': np.array([1, 2, 3]),
            'param2': np.array([4, 5, 6])
        }

        result = mock_client.get_parameters(GetParametersIns(config={}))

        assert result.status.code.value == 0  # OK
        assert result.parameters is not None

    def test_set_parameters_workflow(self, mock_client):
        """Test set_parameters in federated workflow."""
        parameters = [np.array([1, 2, 3]), np.array([4, 5, 6])]

        # Should not raise exception
        mock_client.set_parameters(parameters)

        # Verify load_state_dict was called
        mock_client.model.load_state_dict.assert_called_once()

    def test_fit_workflow_simulation(self, mock_client):
        """Test fit workflow with simulation."""
        from flwr.common import FitIns, Parameters

        # Mock fit inputs
        mock_parameters = Parameters([np.array([1, 2, 3])], "numpy")
        fit_ins = FitIns(
            parameters=mock_parameters,
            config={
                "local_epochs": 1,
                "batch_size": 4,
                "learning_rate": 1e-4
            }
        )

        # Mock the training process
        with patch.object(mock_client, '_simulate_training_step', return_value=0.5):
            result = mock_client.fit(fit_ins)

            assert result.status.code.value == 0  # OK
            assert result.num_examples > 0
            assert "loss" in result.metrics

    def test_evaluate_workflow_simulation(self, mock_client):
        """Test evaluate workflow with simulation."""
        from flwr.common import EvaluateIns, Parameters

        # Mock evaluate inputs
        mock_parameters = Parameters([np.array([1, 2, 3])], "numpy")
        evaluate_ins = EvaluateIns(
            parameters=mock_parameters,
            config={}
        )

        # Mock the evaluation process
        with patch.object(mock_client, '_simulate_validation_step', return_value=(0.3, 3)):
            result = mock_client.evaluate(evaluate_ins)

            assert result.status.code.value == 0  # OK
            assert result.num_examples > 0
            assert "loss" in result.metrics
            assert "action_accuracy" in result.metrics
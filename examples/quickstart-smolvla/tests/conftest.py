"""Pytest configuration and fixtures for SmolVLA tests."""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add the smolvla_example directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "smolvla_example"))


@pytest.fixture
def mock_torch():
    """Mock torch module for testing."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    mock.device.return_value = "cpu"
    return mock


@pytest.fixture
def mock_transformers():
    """Mock transformers module for testing."""
    mock = MagicMock()
    mock.AutoModelForVision2Seq.from_pretrained.return_value = Mock()
    mock.AutoProcessor.from_pretrained.return_value = Mock()
    return mock


@pytest.fixture
def mock_federated_dataset():
    """Mock federated dataset for testing."""
    mock = MagicMock()
    mock.load_partition.return_value = [Mock()]  # Mock dataset partition
    return mock


@pytest.fixture
def mock_dataloader():
    """Mock DataLoader for testing."""
    mock = MagicMock()
    mock.__iter__.return_value = [Mock()]  # Mock batch
    return mock


@pytest.fixture
def sample_client_config():
    """Sample client configuration for testing."""
    return {
        "model_name": "lerobot/smolvla_base",
        "device": "cpu",
        "partition_id": 0,
        "num_partitions": 2
    }


@pytest.fixture
def client_config():
    """Default client configuration for integration tests."""
    return {
        "model_name": "lerobot/smolvla_base",
        "device": "cpu",
        "partition_id": 0,
        "num_partitions": 2
    }


@pytest.fixture
def mock_flower_client():
    """Mock Flower client components."""
    mock_client = MagicMock()
    mock_client.get_parameters.return_value = MagicMock()
    mock_client.set_parameters.return_value = None
    mock_client.fit.return_value = MagicMock()
    mock_client.evaluate.return_value = MagicMock()
    return mock_client


@pytest.fixture(autouse=True)
def mock_imports(monkeypatch, mock_torch, mock_transformers, mock_federated_dataset, mock_dataloader):
    """Automatically mock external dependencies."""
    # Mock torch module
    monkeypatch.setattr("smolvla_example.client_app.torch", mock_torch)
    monkeypatch.setattr("smolvla_example.client_app.torch.cuda", mock_torch.cuda)
    monkeypatch.setattr("smolvla_example.client_app.torch.device", mock_torch.device)
    monkeypatch.setattr("smolvla_example.client_app.torch.utils.data.DataLoader", mock_dataloader)

    # Mock transformers module
    monkeypatch.setattr("smolvla_example.client_app.transformers", mock_transformers)
    monkeypatch.setattr("smolvla_example.client_app.transformers.AutoModelForVision2Seq", mock_transformers.AutoModelForVision2Seq)
    monkeypatch.setattr("smolvla_example.client_app.transformers.AutoProcessor", mock_transformers.AutoProcessor)

    # Mock federated dataset utilities
    monkeypatch.setattr("smolvla_example.client_app.LeRobotDatasetPartitioner", Mock)
    monkeypatch.setattr("smolvla_example.client_app.FederatedLeRobotDataset", Mock(return_value=mock_federated_dataset))


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for tests."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(scope="session")
def test_logger():
    """Test logger fixture."""
    import logging
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    return logger
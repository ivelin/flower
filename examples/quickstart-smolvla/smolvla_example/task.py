"""smolvla_example: A Flower / SmolVLA app for SO-100 robotics."""

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import numpy
import numpy as np
import torch
# Import SmolVLA directly from local repo since it's not in official release yet
import sys
sys.path.insert(0, '/home/ivelin/zk0/lerobot/src')
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
from lerobot.common.datasets.utils import hf_transform_to_torch, get_hf_dataset_safe_version
from torch.utils.data import DataLoader

from datasets.utils.logging import disable_progress_bar

from .lerobot_federated_dataset import FilteredLeRobotDataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import GroupedNaturalIdPartitioner


disable_progress_bar()
fds = {}  # Cache FederatedDataset per task


def get_delta_timestamps():
    # Set up the dataset.
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
        ],
    }
    return delta_timestamps


def get_dataset(task_name="pick_place"):
    # Map task names to dataset repos
    task_to_repo = {
        "pick_place": "lerobot/svla_so100_pick_place",
        "stacking": "lerobot/svla_so100_stacking",
        "folding": "lerobot/svla_so100_folding",
        "pouring": "lerobot/svla_so100_pouring"
    }
    repo_id = task_to_repo.get(task_name, "lerobot/svla_so100_pick_place")
    dataset = LeRobotDataset(repo_id, delta_timestamps=get_delta_timestamps())
    return dataset


def load_data(
    partition_id: int, num_partitions: int, model_name: str, device=None
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Load SO-100 task data for FL"""
    # Map partition_id to task
    task_map = {0: "pick_place", 1: "stacking", 2: "folding", 3: "pouring"}
    task_name = task_map.get(partition_id, "pick_place")
    task_repo = f"lerobot/svla_so100_{task_name}"

    # Only initialize `FederatedDataset` once per task
    global fds
    if fds is None:
        fds = {}

    if task_name not in fds:
        # Each client gets the full dataset for their task (no intra-task partitioning)
        # Since datasets are small (~100 episodes), we use the full dataset
        safe_version = get_hf_dataset_safe_version(task_repo, CODEBASE_VERSION)
        fds[task_name] = FederatedDataset(
            dataset=task_repo,
            partitioners={"train": None},  # No partitioning, use full dataset
            revision=safe_version,
        )

    # Load the full dataset for this task
    partition = fds[task_name].load_partition(0)  # Only one partition per task
    partition.set_transform(hf_transform_to_torch)
    data = FilteredLeRobotDataset(
        repo_id=task_repo,
        hf_dataset=partition,
        delta_timestamps=get_delta_timestamps(),
    )
    # Create dataloader for offline training.
    trainloader = torch.utils.data.DataLoader(
        data,
        num_workers=4,
        batch_size=16,  # Smaller batch size for SmolVLA
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # For evaluation, we'll use held-out episodes from the same dataset
    # No separate testloader needed as in PushT

    return trainloader


def get_model(dataset_stats: dict):
    # Set up the SmolVLA policy.
    # For SmolVLA, we can load the pretrained model and fine-tune it
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", dataset_stats=dataset_stats)
    return policy


def get_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model, parameters) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train(net=None, trainloader=None, epochs=None, device=None) -> None:
    # how frequently (train steps) to print train progress log
    log_freq = 250

    # in lerobot terminology policy is the neural network
    policy = net
    policy.train()
    # policy.to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-6, weight_decay=1e-4)

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in trainloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0 and step > 0:
                print(f"train step: {step} train loss: {loss.item():.3f}")
            step += 1
            assert isinstance(
                epochs, int
            ), f"epochs value: {epochs} , type: {epochs.__class__}"
            if step >= epochs:
                done = True
                break


def test(partition_id: int, net, device, output_dir: Path) -> tuple[Any | float, Any]:
    # For FL simulation, evaluate on held-out episodes from the dataset
    # Map partition_id to task
    task_map = {0: "pick_place", 1: "stacking", 2: "folding", 3: "pouring"}
    task_name = task_map.get(partition_id, "pick_place")
    task_repo = f"lerobot/svla_so100_{task_name}"

    # Load evaluation dataset (use a subset for testing)
    eval_dataset = LeRobotDataset(task_repo, delta_timestamps=get_delta_timestamps(), split="train")
    # Use last 20% of episodes for evaluation
    num_eval_episodes = max(1, int(0.2 * eval_dataset.num_episodes))
    eval_episodes = eval_dataset.episode_data_index[-num_eval_episodes:]

    policy = net
    policy.eval()
    policy.to(device)

    total_loss = 0.0
    total_accuracy = 0.0
    num_samples = 0

    for episode_idx in eval_episodes:
        episode_data = eval_dataset[episode_idx]

        # Get the first observation and action sequence
        obs_image = episode_data["observation.image"][0:2]  # First 2 frames
        obs_state = episode_data["observation.state"][0:2]
        target_action = episode_data["action"][0]  # First action

        # Prepare for policy
        obs_image = obs_image.to(device).unsqueeze(0)  # Add batch dim
        obs_state = obs_state.to(device).unsqueeze(0)

        observation = {
            "observation.image": obs_image,
            "observation.state": obs_state,
        }

        # Predict action
        with torch.inference_mode():
            pred_action = policy.select_action(observation)

        # Calculate loss (MSE between predicted and target action)
        loss = torch.nn.functional.mse_loss(pred_action.squeeze(0), target_action.to(device))
        total_loss += loss.item()

        # For accuracy, use a threshold (simple binary: close enough or not)
        action_diff = torch.abs(pred_action.squeeze(0) - target_action.to(device))
        accuracy = (action_diff < 0.1).float().mean().item()  # 10% tolerance
        total_accuracy += accuracy

        num_samples += 1

    avg_loss = total_loss / num_samples if num_samples > 0 else 1.0
    avg_accuracy = total_accuracy / num_samples if num_samples > 0 else 0.0

    print(f"Evaluation on {task_name}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")

    # Save evaluation results
    eval_dir = output_dir / f"client_{partition_id}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_file = eval_dir / "eval_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
        f.write(f"Num Samples: {num_samples}\n")

    return avg_loss, avg_accuracy

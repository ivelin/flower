"""huggingface_example: A Flower / Hugging Face LeRobot app."""

import warnings

import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from transformers import logging
from logging import INFO

from flwr.common.logger import log
from lerobot_example.task import (
    train,
    test,
    load_data,
    set_params,
    get_params,
    get_model,
    get_output_dir
)

warnings.filterwarnings("ignore", category=FutureWarning)

# To mute warnings reminding that we need to train the model to a downstream task
# This is something this example does.
logging.set_verbosity_error()


# Flower client
class LeRobotClient(NumPyClient):
    def __init__(self, partition_id, model_name, local_epochs, trainloader, nn_device=None, output_dir=None) -> None:
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.net = get_model(model_name=model_name, dataset=trainloader.dataset)
        self.local_epochs = local_epochs
        policy = self.net
        self.device = nn_device
        self.output_dir = output_dir
        if self.device == torch.device("cpu"):
            # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
            policy.diffusion.num_inference_steps = 10        
        policy.to(self.device)

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        set_params(self.net, parameters)
        train(partition_id=self.partition_id, net=self.net, trainloader=self.trainloader, epochs=self.local_epochs, device=self.device, output_dir=self.output_dir)
        return get_params(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, float]]:
        set_params(self.net, parameters)
        loss, accuracy= test(partition_id=self.partition_id, net=self.net, device=self.device, output_dir=self.output_dir)
        testset_len = 1 # we test on one gym generated task
        return float(loss), testset_len, {"accuracy": accuracy}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    log(INFO, f"partition_id={partition_id}, num_partitions={num_partitions}")
    # Discover device  
    nn_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]
    local_epochs = int(context.run_config["local-epochs"])
    log(INFO, f"local_epochs={local_epochs}")
    trainloader = load_data(partition_id, num_partitions, model_name, device=nn_device)
    output_dir = get_output_dir()

    return LeRobotClient(partition_id=partition_id, model_name=model_name, local_epochs=local_epochs, trainloader=trainloader, nn_device=nn_device, output_dir=output_dir).to_client()

app = ClientApp(client_fn=client_fn)

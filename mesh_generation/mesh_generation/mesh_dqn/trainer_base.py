"""Trainer base class for mesh generation."""

from mesh_generation.mesh_dqn.DeepQEnvironSetup import DeepQEnvironSetup
from mesh_generation.mesh_dqn.data_handler import DataHandler
from mesh_generation.mesh_dqn.parameter_server import ParameterServer
from mesh_generation.mesh_dqn.config import load_config
import torch
import numpy as np

class BaseTrainer:
    """Base trainer class for mesh generation."""

    def __init__(self, prefix: str, restart=True):
        self.config = load_config(restart=restart, prefix=prefix)
        self.deep_q_environment_setup = DeepQEnvironSetup(self.config)
        self.parameter_server = ParameterServer(self.config, self.deep_q_environment_setup)
        self.data_handler = DataHandler(self.config)

    def save_models_and_state(self):
        """Save the progress of models being trained."""
        self.data_handler.write()
        torch.save(self.parameter_server.actor_policy_net_1.state_dict(),
                   "./{}/{}actor_policy_net_1.pt".format(self.config.save_dir, self.config.agent_params.prefix))
        torch.save(self.parameter_server.actor_policy_net_2.state_dict(),
                   "./{}/{}actor_policy_net_2.pt".format(self.config.save_dir, self.config.agent_params.prefix))


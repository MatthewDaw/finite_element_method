
import random
import numpy as np

import torch
import os
import sys
import yaml
#from ray import tune, air, train
#from ray.air import session, Checkpoint
from tqdm import tqdm
import time
from itertools import count
import random
import numpy as np
from torch import optim
from matplotlib import pyplot as plt
from collections import namedtuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import math
from shapely.geometry import Polygon
from gym import Env, spaces
from torch import nn
from mesh_generation.mesh_dqn.config import load_config
from mesh_generation.mesh_dqn.parameter_server import ParameterServer
from mesh_generation.mesh_dqn.replay_memory import ReplayMemory
from mesh_generation.mesh_dqn.data_handler import DataHandler
from mesh_generation.mesh_dqn.DeepQEnvironSetup import DeepQEnvironSetup

random.seed(42)
np.random.seed(42)

class Trainer:

    def __init__(self):
        self.config = load_config(restart=False)
        self.transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
        self.reply_memory = ReplayMemory(self.transition, 10000)
        self.criterion = nn.HuberLoss()
        self.parameter_server = ParameterServer(self.config)
        self.data_handler = DataHandler(self.config)
        self.deep_q_environment_setup = DeepQEnvironSetup(self.config)

    def optimize_model(self, optimizer):
        ps = self.parameter_server
        # memory = self.reply_memory
        # flow_config = self.config
        # Transition = self.transition
        # criterion = self.criterion
        # data_handler = self.data_handler
        # if len(memory) < self.config.optimizer.batch_size:
        #     return
        # transitions = memory.sample(self.config.optimizer.batch_size)
        # batch = self.transition(*zip(*transitions))
        #
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                         batch.next_state)), dtype=torch.bool)
        # non_final_next_states = [s for s in batch.next_state if s is not None]

        # state_batch = batch.state
        # action_batch = torch.cat(batch.action).to(self.config.device)
        # reward_batch = torch.cat(batch.reward).to(self.config.device)




    def train(self):
        first = True
        optimizer = self.parameter_server.optimizer_fn(self.parameter_server.policy_net_1.parameters())
        start_ep = len(self.data_handler.rewards) if (self.config.restart) else 0
        previous_state = self.deep_q_environment_setup.get_state()
        for episode in range(start_ep, self.config.agent_params.episodes):
            episode_actions = []
            episode_rewards = []
            acc_rew = 0.0
            self.deep_q_environment_setup.reset()

            for t in tqdm(count()):

                choice, coordinates = self.parameter_server.select_action(previous_state)
                reward, done = self.deep_q_environment_setup.step(choice.item(), coordinates.cpu().detach().numpy())
                state = self.deep_q_environment_setup.get_state()
                episode_actions.append((choice, coordinates))
                episode_rewards.append(reward)
                acc_rew += reward
                reward = torch.tensor([reward])
                previous_state = state

                self.reply_memory.push(previous_state.to(self.config.device), (choice, coordinates),
                                       state.to(self.config.device), reward.to(self.config.device))

                if done:
                    next_state = None
                    break
                self.optimize_model(optimizer)
            print("think more here")

#             namedtuple('Transition',('state', 'action', 'next_state', 'reward'))




if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    print("Training complete.")

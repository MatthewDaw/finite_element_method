
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
from mesh_generation.mesh_dqn.pydantic_objects import Transition, BatchedTransition
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
        if len(self.reply_memory) < self.config.optimizer.batch_size:
            return

        transitions = self.reply_memory.sample(self.config.optimizer.batch_size)
        print("think more here")

        batched_transition = BatchedTransition(
        state=[t.state for t in transitions],
        detached_state_choice_output=[t.state_choice_output for t in transitions],
        next_state=[t.next_state for t in transitions],
        reward=[t.reward for t in transitions],
    )

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batched_transition.next_state)), dtype=torch.bool)
        non_final_next_states = [s for s in batched_transition.next_state if s is not None]

        # Get batch
        state_batch = batched_transition.state
        action_batch = torch.cat(batched_transition.state_choice_output).to(self.config.device)
        reward_batch = torch.cat(batched_transition.reward).to(self.config.device)

        # Easiest way to batch this
        loader = DataLoader(state_batch, batch_size=self.config.optimizer.batch_size)
        for data in loader:
            output = ps.policy_net_1(data)

        state_action_values = output[:, action_batch[:, 0]].diag()

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.config.optimizer.batch_size).to(self.config.device).float()
        loader = DataLoader(non_final_next_states, batch_size=self.config.optimizer.batch_size)

        # get batched output
        for data in loader:
            try:
                output = ps.policy_net_2(data).max(1)[0]
            except RuntimeError:
                print("\n\n")
                # print(data)
                # print(data.x)
                # print(data.edge_index)
                # print(data.edge_attr)
                print("\n\n")
                raise


        next_state_values[non_final_mask] = output
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config.epsilon.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values.float(), expected_state_action_values.float()).float()
        self.data_handler.losses.append(loss.item())
        if ((len(self.reply_memory) % 25) == 0):
            np.save("./{}/{}losses.npy".format(self.config.save_dir, self.config.agent_params.prefix),
                    self.data_handler.losses)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("think more here")

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

        for episode in range(start_ep, self.config.agent_params.episodes):
            episode_actions = []
            episode_rewards = []
            acc_rew = 0.0
            self.deep_q_environment_setup.reset()
            previous_state = self.deep_q_environment_setup.get_state()

            for t in tqdm(count()):
                state_choice_output = self.parameter_server.select_action(previous_state)
                detached_state_choice_output = state_choice_output.detach().numpy()

                reward = self.deep_q_environment_setup.step(detached_state_choice_output)
                episode_actions.append(detached_state_choice_output)
                episode_rewards.append(reward)
                acc_rew += reward
                reward = torch.tensor([reward])

                if not self.deep_q_environment_setup.terminated:
                    next_state = None
                    transition = Transition(
                        state=previous_state.to(self.config.device),
                        detached_state_choice_output=state_choice_output.detach(),
                        next_state=next_state,
                        reward=reward.to(self.config.device)
                    )
                else:
                    next_state = self.deep_q_environment_setup.get_state()
                    transition = Transition(
                        state=previous_state.to(self.config.device),
                        state_choice_output=state_choice_output.detach(),
                        next_state=next_state.to(self.config.device),
                        reward=reward.to(self.config.device)
                    )

                self.reply_memory.push(transition)

                self.optimize_model(optimizer)

                if self.deep_q_environment_setup.terminated:
                    self.data_handler.ep_rewards.append(acc_rew)
                    break
            self.data_handler.all_actions.append(np.array(episode_actions))
            self.data_handler.all_rewards.append(np.array(episode_rewards))



            if ((episode % self.config.agent_params.target_update) == 0):
                # target_net.load_state_dict(policy_net.state_dict())
                if (first):
                    optimizer = self.parameter_server.optimizer_fn(self.parameter_server.policy_net_1.parameters())
                    first = False
                else:
                    optimizer = self.parameter_server.optimizer_fn(self.parameter_server.policy_net_2.parameters())
                    first = True


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    print("Training complete.")

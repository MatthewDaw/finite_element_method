"""MeshGeneratorTrainer for mesh point placement algorithm."""

import torch
import random
import numpy as np
from collections import namedtuple
from torch_geometric.loader import DataLoader
from torch import nn
from mesh_generation.mesh_dqn.pydantic_objects import Transition, BatchedTransition
from mesh_generation.mesh_dqn.replay_memory import ReplayMemory
from mesh_generation.mesh_dqn.data_handler import DataHandler

import warnings

from mesh_generation.mesh_dqn.trainer_base import BaseTrainer

# Suppress specific warnings by matching the message
warnings.filterwarnings(
    "ignore",
    message="Initializing InteriorFacetBasis\\(MeshTri1, ElementTriP1\\) with no facets.",
)
warnings.filterwarnings(
    "ignore",
    message=r"Criterion not satisfied in \d+ refinement loops\.",
)
from scipy.sparse import SparseEfficiencyWarning

# random.seed(42)
# np.random.seed(42)

class MeshGeneratorTrainer(BaseTrainer):

    def __init__(self):
        super().__init__("mesh_generator")
        self.use_model_1 = True
        self.reply_memory = ReplayMemory(10000)
        self.criterion = nn.HuberLoss()
        self.data_handler = DataHandler(self.config)

    def optimize_model(self):
        """Optimize the model."""

        if len(self.reply_memory) < self.config.optimizer.batch_size:
            return

        transitions = self.reply_memory.sample(self.config.optimizer.batch_size)
        batched_transition = BatchedTransition(
            state=[t.state for t in transitions],
            state_choice_output=[t.state_choice_output for t in transitions],
            next_state=[t.next_state for t in transitions],
            reward=[t.reward for t in transitions],
        )

        # Easiest way to batch this
        loader = DataLoader(batched_transition.state, batch_size=self.config.optimizer.batch_size)
        for data in loader:
            action_output = self.parameter_server.actor_policy_net_1(data)
            current_q_output = self.parameter_server.critic_net_1(data, action_output)

        target_q_outputs = torch.zeros(current_q_output.shape[0]).to(self.config.device).float()
        non_final_next_states_mask = [s is not None for s in batched_transition.next_state]
        non_final_next_states = [s for s in batched_transition.next_state if s is not None]

        if len(non_final_next_states) > 0:
            loader = DataLoader(non_final_next_states, batch_size=self.config.optimizer.batch_size)
            for data in loader:
                action_output = self.parameter_server.actor_policy_net_2(data)
                estimated_q = self.parameter_server.critic_net_2(data, action_output)[:,0]
                target_q_outputs[non_final_next_states_mask] = estimated_q

        target_q_outputs = torch.tensor(batched_transition.reward).to(self.config.device) + self.config.epsilon.gamma * target_q_outputs
        critic_loss = self.criterion(current_q_output.float(), target_q_outputs.float())
        # Update critic network
        self.parameter_server.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.parameter_server.critic_optimizer.step()

        loader = DataLoader(batched_transition.state, batch_size=self.config.optimizer.batch_size)
        for data in loader:
            action_output = self.parameter_server.actor_policy_net_1(data)
            current_q_output = self.parameter_server.critic_net_1(data, action_output)

        # Actor Loss: Use critic to evaluate actor's output for the current state
        actor_loss = -torch.mean(current_q_output)

        # Optimize the model
        self.parameter_server.optimizer.zero_grad()
        actor_loss.backward()
        self.parameter_server.optimizer.step()

        self.data_handler.losses.append(actor_loss.item())
        self.data_handler.critic_losses.append(critic_loss.item())
        if ((len(self.reply_memory) % 25) == 0):
            np.save("./{}/{}losses.npy".format(self.config.save_dir, self.config.agent_params.prefix),
                    self.data_handler.losses)
            np.save("./{}/{}critic_losses.npy".format(self.config.save_dir, self.config.agent_params.prefix),
                    self.data_handler.critic_losses)
            print("Losses saved.")
            print(f"Losess: {self.data_handler.losses[-25:]}")


    def train(self):
        start_ep = len(self.data_handler.rewards) if (self.config.restart) else 0

        for episode in range(start_ep, self.config.agent_params.episodes):
            episode_actions = []
            episode_rewards = []
            acc_rew = 0.0
            self.deep_q_environment_setup.reset()
            previous_state = self.deep_q_environment_setup.get_state()

            while True:
                state_choice_output = self.parameter_server.select_action(previous_state, episode)
                detached_state_choice_output = state_choice_output.detach().numpy()
                reward = self.deep_q_environment_setup.step(detached_state_choice_output)
                episode_actions.append(detached_state_choice_output)
                episode_rewards.append(reward)
                acc_rew += reward
                reward = torch.tensor([reward])

                if self.deep_q_environment_setup.terminated:
                    next_state = None
                    transition = Transition(
                        state=previous_state.to(self.config.device),
                        state_choice_output=state_choice_output.detach(),
                        next_state=next_state,
                        reward=reward.to(self.config.device)
                    )
                else:

                    next_state = self.deep_q_environment_setup.get_state()
                    transition = Transition(
                        state=previous_state.to(self.config.device),
                        state_choice_output=state_choice_output.to(self.config.device),
                        next_state=next_state.to(self.config.device),
                        reward=reward.to(self.config.device)
                    )
                    previous_state = next_state

                self.reply_memory.push(transition)

                self.optimize_model()

                if self.deep_q_environment_setup.terminated:
                    self.data_handler.ep_rewards.append(acc_rew)
                    break

            self.data_handler.shape_parameters.append(np.stack([self.deep_q_environment_setup.course_problem_setup.shape_parameters.x_points[0], self.deep_q_environment_setup.course_problem_setup.shape_parameters.y_points[0]]).shape)
            self.data_handler.all_actions.append(np.array(episode_actions))
            self.data_handler.all_rewards.append(np.array(episode_rewards))

            if ((episode % self.config.agent_params.target_update) == 0):
                if self.use_model_1:
                    self.parameter_server.optimizer = self.parameter_server.optimizer_fn(self.parameter_server.actor_policy_net_1.parameters())
                    self.parameter_server.critic_optimizer = self.parameter_server.optimizer_fn(self.parameter_server.critic_net_1.parameters())
                else:
                    self.parameter_server.optimizer = self.parameter_server.optimizer_fn(self.parameter_server.actor_policy_net_2.parameters())
                    self.parameter_server.critic_optimizer = self.parameter_server.optimizer_fn(self.parameter_server.critic_net_2.parameters())
                self.use_model_1 = not(self.use_model_1)

            if (len(self.data_handler.ep_rewards) % 15 == 0):
                self.save_models()

if __name__ == '__main__':
    trainer = MeshGeneratorTrainer()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Initializing InteriorFacetBasis\\(MeshTri1, ElementTriP1\\) with no facets."
        )
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        warnings.filterwarnings("ignore",
                                message="Robust predicates not available, falling back on non-robust implementation")
        trainer.train()
    print("Training complete.")

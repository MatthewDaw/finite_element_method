
import torch
from mesh_generation.mesh_dqn.node_setting_gcnn import NodeSettingNet, CriticNet
from torch import optim
import numpy as np

from mesh_generation.mesh_dqn.pydantic_objects import FlowConfig
from mesh_generation.simple_node_setter import PointGeneratorGCN


class CustomLRScheduler:
    def __init__(self, optimizer, decrement=1e-6, min_lr=0):
        self.optimizer = optimizer
        self.decrement = decrement
        self.min_lr = min_lr

    def step(self):
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group['lr'] - self.decrement, self.min_lr)
            param_group['lr'] = new_lr


class ParameterServer:
    def __init__(self, config: FlowConfig, deep_q_environment_setup):
        self.deep_q_environment_setup = deep_q_environment_setup
        self.config = config
        self.save_dir = config.save_dir
        self.PREFIX = config.agent_params.prefix

        self.max_episode_to_do_random_actions = max(int(config.agent_params.episodes * 0.1), 10000)

        # output parameters
        # nodes 1-3 are decision nodes
        # 1 if one is biggest, do nothing
        # 2 if is biggest, add a node
        # 3 if is biggest, remove a node
        # 4 is coordinate of node to add or remove (normalized from 0 to 1)
        # 5 is y coordinate of node to add or remove (normalized from 0 to 1)

        self.actor_policy_net_1 = PointGeneratorGCN(13, 120, 8).to(
            self.config.device).float()
        self.actor_policy_net_2 = PointGeneratorGCN(13, 120, 8).to(
            self.config.device).float()

        # self.actor_policy_net_1 = NodeSettingNet(config.agent_params.output_dim_size, conv_width=128, topk=0.1).to(
        #     self.config.device).float()
        # self.actor_policy_net_2 = NodeSettingNet(config.agent_params.output_dim_size, conv_width=128, topk=0.1).to(
        #     self.config.device).float()

        # self.actor_policy_net_1.set_num_nodes(config.agent_params.NUM_INPUTS)
        # self.actor_policy_net_2.set_num_nodes(config.agent_params.NUM_INPUTS)

        self.critic_net_1 = CriticNet(config.agent_params.output_dim_size)
        self.critic_net_1.set_num_nodes(config.agent_params.NUM_INPUTS)
        self.critic_net_2 = CriticNet(config.agent_params.output_dim_size)
        self.critic_net_2.set_num_nodes(config.agent_params.NUM_INPUTS)

        if not config.restart:
            for i in range(config.restart_num-1):
                self.PREFIX = "restart_" + self.PREFIX
            self.actor_policy_net_1.load_state_dict(torch.load(
                                    "./{}/{}actor_policy_net_1.pt".format(config.save_dir, config.agent_params.prefix)))
            self.actor_policy_net_2.load_state_dict(torch.load(
                                    "./{}/{}actor_policy_net_2.pt".format(config.save_dir, config.agent_params.prefix)))
            self.PREFIX = "restart_" + self.PREFIX

        self.optimizer_fn = lambda parameters: optim.Adam(parameters, lr=float(config.optimizer.lr),
                                                          weight_decay=float(config.optimizer.weight_decay))
        self.optimizer = self.optimizer_fn(self.actor_policy_net_1.parameters())
        self.critic_optimizer = self.optimizer_fn(self.critic_net_1.parameters())
        self.scheduler = CustomLRScheduler(self.optimizer, decrement=1e-6, min_lr=1e-5)
        self.num_grads = 0
        self.select = True

    def apply_gradients(self, *gradients):

        if((self.num_grads % int(self.config.agent_params.target_update)) == 0):
            self.select = not(self.select)

        #self.optimizer.zero_grad()
        self.optimizer.step()
        self.scheduler.step()
        self.num_grads += 1

        if(self.select):
            self.actor_policy_net_1.set_gradients(gradients[0])
            self.optimizer = self.optimizer_fn(self.actor_policy_net_1.parameters())
            return self.actor_policy_net_1.state_dict()
        else:
            self.actor_policy_net_2.set_gradients(gradients[0])
            self.optimizer = self.optimizer_fn(self.actor_policy_net_2.parameters())
            return self.actor_policy_net_2.state_dict()

    def state_dict(self):
        if(self.select):
            return self.actor_policy_net_1.state_dict()
        else:
            return self.actor_policy_net_2.state_dict()

    def choose_random_action(self):
        chosen_point1 = self.deep_q_environment_setup.select_random_point_in_mesh()
        chosen_point2 = self.deep_q_environment_setup.select_random_point_in_mesh()
        stacked_points = np.array([[chosen_point1.x, chosen_point1.y], [chosen_point2.x, chosen_point2.y]])
        scaled_points = self.deep_q_environment_setup.perform_scaling(stacked_points)
        choice_vector = np.zeros(4)
        random_action = np.random.random()
        random_action = 0.1
        # terminate
        if random_action < 0.02:
            choice_vector[0] = 1
        # add a point
        elif random_action < 0.8:
            choice_vector[1] = 1
        # remove a point
        elif random_action < 0.9:
            choice_vector[2] = 1
        # both add and remove
        else:
            choice_vector[3] = 0
        return torch.tensor(np.concat([choice_vector, scaled_points[0], scaled_points[1]]))


    def select_action(self, state, episode_number):
        if np.random.random() > episode_number / self.max_episode_to_do_random_actions:
            return self.choose_random_action()
        state_choice_output = self.actor_policy_net_1(state)
        return state_choice_output[0]

    def select(self):
        return self.select

    def write(self):
        torch.save(self.actor_policy_net_1.state_dict(),
                    "/Users/matthewdaw/Documents/fem/MeshDQN/{}/{}actor_policy_net_1.pt".format(self.save_dir, self.PREFIX))
        torch.save(self.actor_policy_net_2.state_dict(),
                    "/Users/matthewdaw/Documents/fem/MeshDQN/{}/{}actor_policy_net_2.pt".format(self.save_dir, self.PREFIX))



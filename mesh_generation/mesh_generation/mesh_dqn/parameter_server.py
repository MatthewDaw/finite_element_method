
import torch
from mesh_generation.mesh_dqn.node_setting_gcnn import NodeSettingNet
from torch import optim

from mesh_generation.mesh_dqn.pydantic_objects import FlowConfig


class ParameterServer:
    def __init__(self, config: FlowConfig):
        self.config = config
        self.save_dir = config.save_dir
        self.PREFIX = config.agent_params.prefix

        # output parameters
        # nodes 1-3 are decision nodes
        # 1 if one is biggest, do nothing
        # 2 if is biggest, add a node
        # 3 if is biggest, remove a node
        # 4 is coordinate of node to add or remove (normalized from 0 to 1)
        # 5 is y coordinate of node to add or remove (normalized from 0 to 1)


        self.policy_net_1 = NodeSettingNet(config.agent_params.output_dim_size, conv_width=128, topk=0.1).to(
            self.config.device).float()
        self.policy_net_2 = NodeSettingNet(config.agent_params.output_dim_size, conv_width=128, topk=0.1).to(
            self.config.device).float()

        self.policy_net_1.set_num_nodes(config.agent_params.NUM_INPUTS)
        self.policy_net_2.set_num_nodes(config.agent_params.NUM_INPUTS)

        if config.restart:
            for i in range(config.restart_num-1):
                self.PREFIX = "restart_" + self.PREFIX
            self.policy_net_1.load_state_dict(torch.load(
                                    "./{}/{}policy_net_1.pt".format(config.save_dir, config.agent_params.prefix)))
            self.policy_net_2.load_state_dict(torch.load(
                                    "./{}/{}policy_net_2.pt".format(config.save_dir, config.agent_params.prefix)))
            self.PREFIX = "restart_" + self.PREFIX

        self.optimizer_fn = lambda parameters: optim.Adam(parameters, lr=float(config.optimizer.lr),
                                                          weight_decay=float(config.optimizer.weight_decay))
        self.optimizer = self.optimizer_fn(self.policy_net_1.parameters())
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                       milestones=[500000, 1000000, 1500000], gamma=0.1)
        if config.restart:
            for i in range(449129):
                self.scheduler.step()

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
            self.policy_net_1.set_gradients(gradients[0])
            self.optimizer = self.optimizer_fn(self.policy_net_1.parameters())
            return self.policy_net_1.state_dict()
        else:
            self.policy_net_2.set_gradients(gradients[0])
            self.optimizer = self.optimizer_fn(self.policy_net_2.parameters())
            return self.policy_net_2.state_dict()

    def state_dict(self):
        if(self.select):
            return self.policy_net_1.state_dict()
        else:
            return self.policy_net_2.state_dict()

    def select_action(self, state):
        state_choice_output = self.policy_net_1(state)
        return state_choice_output[0]

    def select(self):
        return self.select

    def write(self):
        torch.save(self.policy_net_1.state_dict(),
                    "/Users/matthewdaw/Documents/fem/MeshDQN/{}/{}policy_net_1.pt".format(self.save_dir, self.PREFIX))
        torch.save(self.policy_net_2.state_dict(),
                    "/Users/matthewdaw/Documents/fem/MeshDQN/{}/{}policy_net_2.pt".format(self.save_dir, self.PREFIX))

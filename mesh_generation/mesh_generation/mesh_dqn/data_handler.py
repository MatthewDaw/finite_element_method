
import numpy as np
import matplotlib.pyplot as plt

from mesh_generation.mesh_dqn.pydantic_objects import FlowConfig


def _movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

class DataHandler:
    """Class to handle data for the DQN training."""

    def __init__(self, config: FlowConfig):
        self.config = config
        self.save_dir = self.config.save_dir
        self.rewards = []
        self.ep_rewards = []
        self.losses = []
        self.critic_losses = []
        self.actions = []
        self.epss = []
        self.all_actions = []
        self.all_rewards = []
        self.shape_parameters = []

        if(self.config.restart):

            self.all_actions = list(np.load("./{}/{}actions.npy".format(self.save_dir, config.agent_params.prefix), allow_pickle=True))
            self.all_rewards = list(np.load("./{}/{}rewards.npy".format(self.save_dir, config.agent_params.prefix), allow_pickle=True))

            for i in range(self.config['restart_num']-1):
                self.save_dir += "RESTART_"
            try:
                self.rewards = list(np.load(self.save_dir + "reward.npy", allow_pickle=True))
            except OSError:
                self.rewards = []
            try:
                self.ep_rewards = list(np.load(self.save_dir + "rewards.npy", allow_pickle=True))
            except OSError:
                self.ep_rewards = []
            try:
                self.losses = list(np.load(self.save_dir + "losses.npy", allow_pickle=True))
                self.critic_losses = list(np.load(self.save_dir + "critic_losses.npy", allow_pickle=True))
            except OSError:
                self.losses = []
                self.critic_losses = []
            try:
                self.actions = list(np.load(self.save_dir + "actions.npy", allow_pickle=True))
            except OSError:
                self.actions = []
            try:
                self.epss = list(np.load(self.save_dir + "eps.npy", allow_pickle=True))
            except OSError:
                self.epss = []
            self.save_dir += "RESTART_"
            print("\n\nWRITING\n\n")
            self.write()

    def add_eps(self, eps):
        """Add an epsilon to the data."""
        self.epss.append(eps)

    def num_eps(self):
        """Return the number of episodes."""
        return len(self.epss)

    def add_loss(self, loss):
        """Add a loss to the data."""
        self.losses.append(loss)

    def add_critic_loss(self, loss):
        """Add a loss to the data."""
        self.critic_losses.append(loss)

    def add_episode(self, ep_rew, ep_action):
        """Add an episode to the data."""
        self.rewards.append(sum(ep_rew))
        self.ep_rewards.append(ep_rew)
        self.actions.append(ep_action)

    def write(self):
        """Write the data to disk."""
        np.save(self.save_dir + "reward.npy", self.rewards)
        np.save(self.save_dir + "rewards.npy", self.ep_rewards)
        np.save(self.save_dir + "losses.npy", self.losses)
        np.save(self.save_dir + "critic_losses.npy", self.critic_losses)
        np.save(self.save_dir + "actions.npy", self.actions)
        np.save(self.save_dir + "eps.npy", self.epss)

    def plot(self):
        """Plot the training reward."""
        fig, ax = plt.subplots()
        ax.plot(self.rewards)
        if(len(self.rewards) >= 25):
            ax.plot(list(range(len(self.rewards)))[24:], _movingaverage(self.rewards, 25))

        if(len(self.rewards) >= 200):
            ax.plot(list(range(len(self.rewards)))[199:], _movingaverage(self.rewards, 200))

        ax.set(xlabel="Episode", ylabel="Reward")
        ax.set_title("DQN Training Reward")
        plt.savefig(self.save_dir + "reward.png".format(self.config.save_dir, self.config.agent_params.prefix))
        plt.close()

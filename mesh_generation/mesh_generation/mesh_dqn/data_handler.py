
import numpy as np
import matplotlib.pyplot as plt

from mesh_generation.mesh_dqn.pydantic_objects import FlowConfig
import os

def _movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

#         print(f"""
#         avg_percent_points_in_shape: {avg_percent_points_in_shape}
#         avg_percent_points_within_bounds: {avg_percent_points_within_bounds}
#         avg_average_variance_loss: {avg_average_variance_loss}
#         average_loss {average_loss}
#         self.training_iteration_count: {self.training_iteration_count}
#         """)

class DataHandler:
    """Class to handle data for the DQN training."""

    def __init__(self, config: FlowConfig):
        self.config = config
        self.save_dir = self.config.save_dir
        self.training_iteration_count = 0

        self.avg_percent_points_in_shape = []
        self.avg_percent_points_within_bounds = []
        self.avg_average_variance_loss = []
        self.average_loss = []
        self.training_iteration_count_list = []


        if(not self.config.restart):

            avg_percent_points_in_shape_path = "./{}/{}avg_percent_points_in_shape.npy".format(self.save_dir, config.agent_params.prefix)
            if os.path.exists(avg_percent_points_in_shape_path):
                self.avg_percent_points_in_shape = list(np.load(avg_percent_points_in_shape_path, allow_pickle=True))

            avg_percent_points_within_bounds_path = "./{}/{}avg_percent_points_within_bounds.npy".format(self.save_dir, config.agent_params.prefix)
            if os.path.exists(avg_percent_points_within_bounds_path):
                self.avg_percent_points_within_bounds = list(np.load(avg_percent_points_within_bounds_path, allow_pickle=True))

            avg_average_variance_loss_path = "./{}/{}avg_average_variance_loss.npy".format(self.save_dir, config.agent_params.prefix)
            if os.path.exists(avg_average_variance_loss_path):
                self.avg_average_variance_loss = list(np.load(avg_average_variance_loss_path, allow_pickle=True))

            average_loss_path = "./{}/{}average_loss.npy".format(self.save_dir, config.agent_params.prefix)
            if os.path.exists(average_loss_path):
                self.average_loss = list(np.load(average_loss_path, allow_pickle=True))

            training_iteration_count_list_path = "./{}/{}training_iteration_count_list.npy".format(self.save_dir, config.agent_params.prefix)
            if os.path.exists(training_iteration_count_list_path):
                self.training_iteration_count_list = list(np.load(training_iteration_count_list_path, allow_pickle=True))
                self.training_iteration_count = self.training_iteration_count_list[-1]

            print("\n\nWRITING\n\n")
            self.write()


    def write(self):
        """Write the data to disk."""
        np.save("./{}/{}avg_percent_points_in_shape.npy".format(self.save_dir, self.config.agent_params.prefix),
                self.avg_percent_points_in_shape)
        np.save("./{}/{}avg_percent_points_within_bounds.npy".format(self.save_dir, self.config.agent_params.prefix),
                self.avg_percent_points_within_bounds)
        np.save("./{}/{}avg_average_variance_loss.npy".format(self.save_dir, self.config.agent_params.prefix),
                self.avg_average_variance_loss)
        np.save("./{}/{}average_loss.npy".format(self.save_dir, self.config.agent_params.prefix),
                self.average_loss)
        np.save("./{}/{}training_iteration_count_list.npy".format(self.save_dir, self.config.agent_params.prefix),
                self.training_iteration_count_list)


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

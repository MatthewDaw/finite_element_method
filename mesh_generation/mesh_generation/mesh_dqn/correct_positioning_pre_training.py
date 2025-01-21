"""This module serves to pretrain the model to get it to reliably put points somewhere in the shape."""
import copy

from pycparser.ply.yacc import restart
from torch_geometric.graphgym import optim

from mesh_generation.mesh_dqn.pydantic_objects import NonRLTransition
from mesh_generation.mesh_dqn.replay_memory import ReplayMemory
from mesh_generation.mesh_dqn.trainer_base import BaseTrainer
import torch
import numpy as np

from mesh_generation.mesh_dqn.training_results.policy_switcher import PolicySwitcher
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data as TorchGeoData

class BatchedNonRLTransition(Dataset):
    def __init__(self, state, shape_parameters, scaling_coefficients, shapely_polygon, expected_average_point_variance, points):
        self.state = state
        self.shape_parameters = shape_parameters
        self.scaling_coefficients = scaling_coefficients
        self.shapely_polygon = shapely_polygon
        self.expected_average_point_variance = expected_average_point_variance
        self.points = points

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return {
            "state": self.state[idx],
            "shape_parameters": self.shape_parameters[idx],
            "scaling_coefficients": self.scaling_coefficients[idx],
            "shapely_polygon": self.shapely_polygon[idx],
            "expected_average_point_variance": self.expected_average_point_variance[idx],
            "points": self.points[idx]
        }

def custom_collate_fn(batch):
    # Check if the batch contains `torch_geometric.data.Data` objects
    if isinstance(batch[0], dict):
        batched_data = {}
        for key in batch[0]:
            if isinstance(batch[0][key], TorchGeoData):
                # Use torch_geometric's Batch for Data objects
                batched_data[key] = Batch.from_data_list([item[key] for item in batch])
            else:
                # Default collation for other types
                try:
                    batched_data[key] = np.array([item[key] for item in batch])
                except:
                    batched_data[key] = [item[key] for item in batch]
        return batched_data
    elif isinstance(batch[0], TorchGeoData):
        # Directly batch Data objects
        return Batch.from_data_list(batch)
    else:
        raise TypeError("Batch contains unsupported data types.")

class PositionPretrainer(BaseTrainer):
    """Pretrain the model to get it to reliably put points somewhere in the shape."""

    def __init__(self):
        super().__init__("point_positioning", restart=True)
        self.reply_memory = ReplayMemory(10000000)
        self.allow_shape_distance_optimizing = PolicySwitcher(self.config.correct_positioning_pre_training.successes_needed_to_switch_to_next_policy)
        self.allow_point_spread_optimizing = PolicySwitcher(self.config.correct_positioning_pre_training.successes_needed_to_switch_to_next_policy)


    def optimize_model(self):
        """Optimize the model."""
        expected_average_point_variance = 0

        transitions = self.reply_memory.memory
        batched_transition = BatchedNonRLTransition(
            state=[t.state for t in transitions],
            shape_parameters=[t.shape_parameters for t in transitions],
            scaling_coefficients=[t.shape_transformation_parameters for t in transitions],
            shapely_polygon=[t.shaplely_polygon for t in transitions],
            expected_average_point_variance=[t.expected_average_point_variance for t in transitions],
            points=[t.points for t in transitions]
        )

        loader = DataLoader(
            batched_transition,
            batch_size=self.config.optimizer.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn
        )

        losses = []
        percent_points_within_bounds = []
        percent_points_in_shape = []
        average_variance_loss = []

        iteration = 0
        count = 0
        for batch in loader:

            data = batch["state"]
            shape_parameters = batch["shape_parameters"]
            scaling_coefficients = batch["scaling_coefficients"]
            shapely_polygons = batch["shapely_polygon"]
            points = batch["points"]
            expected_average_point_variance = batch["expected_average_point_variance"]

            # self.deep_q_environment_setup.visualize_debug_with_input(data[2], shapely_polygons[2], scaling_coefficients[2])

            count += 1
            iteration += 1
            loss_for_bad_variance = torch.tensor(0.4, requires_grad=True)
            mean_loss_for_position = torch.tensor(0.0, requires_grad=True)
            violating_loss = torch.tensor(0.0, requires_grad=True)
            state_choice_outputs_unmasked = self.parameter_server.actor_policy_net_1(data)

            min_output = 0
            max_output = 1

            if not self.allow_shape_distance_optimizing.check_if_ready_to_proceed():
                min_output = 0.25
                max_output = 0.75

            # Create masks for the conditions
            negative_mask = (state_choice_outputs_unmasked[:,4:] < min_output).any(dim=1)  # Rows with at least one negative value
            greater_than_one_mask = (state_choice_outputs_unmasked[:,4:] > max_output).any(dim=1)  # Rows with at least one value > 1
            # Combine the masks using logical OR
            combined_mask = negative_mask | greater_than_one_mask
            violating_entries = state_choice_outputs_unmasked[combined_mask]

            percentage_within_bounds = len(state_choice_outputs_unmasked[~combined_mask]) / len(state_choice_outputs)

            if percentage_within_bounds < 0.1 or self.data_handler.training_iteration_count > 200:
                self.allow_shape_distance_optimizing.add_success()
            else:
                self.allow_shape_distance_optimizing.add_failure()

            state_choice_outputs = state_choice_outputs_unmasked

            percent_points_within_bounds.append(percentage_within_bounds)
            if not self.allow_shape_distance_optimizing.check_if_ready_to_proceed():
                below_zero_loss = torch.abs(violating_entries[violating_entries < min_output])  # Negative values
                above_one_loss = violating_entries[violating_entries > max_output] - max_output  # Values exceeding 1
                violating_loss = below_zero_loss.sum() + above_one_loss.sum()
            else:
                loss_for_position = []
                add_point_in_shape_mask = []
                remove_point_in_shape_mask = []
                for row in range(len(state_choice_outputs)):
                    state_choice_output = state_choice_outputs[row]
                    reward, add_point_in_shape, remove_point_in_shape = self.deep_q_environment_setup.calc_simple_reward_for_correct_position(state_choice_output, scaling_coefficients[~combined_mask][row], shapely_polygons[~combined_mask][row], data[~combined_mask][row])
                    loss_for_position.append(reward)
                    add_point_in_shape_mask.append(add_point_in_shape)
                    remove_point_in_shape_mask.append(remove_point_in_shape)
                loss_for_position = torch.stack(loss_for_position)
                mean_loss_for_position = torch.mean(loss_for_position)
                num_points_in_shape = torch.sum(torch.stack(add_point_in_shape_mask)) + torch.sum(torch.stack(remove_point_in_shape_mask))
                percent_points_in_shape.append(num_points_in_shape / (len(state_choice_outputs)*2))
                if percent_points_in_shape[-1] > 0.95:
                    self.allow_point_spread_optimizing.add_success()
                else:
                    self.allow_point_spread_optimizing.add_failure()
                if percent_points_in_shape[-1] == 0:
                    self.allow_point_spread_optimizing.set_massive_failure()

                if self.allow_point_spread_optimizing.check_if_ready_to_proceed() or True:
                    add_points_in_polygon = state_choice_outputs[add_point_in_shape_mask]
                    remove_points_in_polygon = state_choice_outputs[remove_point_in_shape_mask]
                    if len(add_points_in_polygon) > 0 and len(remove_points_in_polygon) > 0:
                        if iteration % 10 == 0:
                            self.parameter_server.scheduler.step()
                        add_points = add_points_in_polygon[:, 4:6]
                        remove_points = remove_points_in_polygon[:, 6:8]
                        point_variance_inside_shape = (self.deep_q_environment_setup.calculate_point_variance(add_points) + self.deep_q_environment_setup.calculate_point_variance(remove_points))/2
                        loss_for_bad_variance = torch.clip(expected_average_point_variance[0] - point_variance_inside_shape, 0, 0.4)
                        average_variance_loss.append(loss_for_bad_variance.item())

            if not self.allow_shape_distance_optimizing.check_if_ready_to_proceed():
                loss = violating_loss
            elif not self.allow_point_spread_optimizing.check_if_ready_to_proceed() and loss_for_bad_variance == 0:
                loss = mean_loss_for_position
            else:
                loss = mean_loss_for_position + loss_for_bad_variance
            if loss > 0:
                self.parameter_server.optimizer.zero_grad()
                loss.backward()
                self.parameter_server.optimizer.step()
                self.parameter_server.actor_policy_net_1.clip_weights()
            losses.append(loss.item())
            self.data_handler.training_iteration_count += 1

            if iteration % 10 == 0:
                avg_percent_points_within_bounds = 0
                if len(percent_points_within_bounds) != 0:
                    avg_percent_points_within_bounds = np.mean(percent_points_within_bounds)

                avg_percent_points_in_shape = 0
                if len(percent_points_in_shape) != 0:
                    avg_percent_points_in_shape = np.mean(percent_points_in_shape)

                avg_average_variance_loss = None
                if len(average_variance_loss) != 0:
                    avg_average_variance_loss = np.mean(average_variance_loss)

                average_loss = np.mean(losses)
                self.data_handler.avg_percent_points_in_shape.append(avg_percent_points_in_shape)
                self.data_handler.avg_percent_points_within_bounds.append(avg_percent_points_within_bounds)
                self.data_handler.avg_average_variance_loss.append(avg_average_variance_loss)
                self.data_handler.average_loss.append(average_loss)
                self.data_handler.training_iteration_count_list.append(self.data_handler.training_iteration_count)

        avg_percent_points_within_bounds = 0
        if len(percent_points_within_bounds) != 0:
            avg_percent_points_within_bounds = np.mean(percent_points_within_bounds)

        avg_percent_points_in_shape = 0
        if len(percent_points_in_shape) != 0:
            avg_percent_points_in_shape = np.mean(percent_points_in_shape)

        avg_average_variance_loss = None
        if len(average_variance_loss) != 0:
            avg_average_variance_loss = np.mean(average_variance_loss)

        average_loss = np.mean(losses)

        # self.data_handler.training_iteration_count
        print(f"""
        avg_percent_points_in_shape: {avg_percent_points_in_shape}
        avg_percent_points_within_bounds: {avg_percent_points_within_bounds}
        avg_average_variance_loss: {avg_average_variance_loss}
        average_loss {average_loss}
        self.training_iteration_count: {self.data_handler.training_iteration_count}
        """)


    def train(self):
        """Train the model."""
        start_batch = 0
        for batch_set in range(start_batch, self.config.correct_positioning_pre_training.num_batches_to_train_for):
            crashed = False
            self.deep_q_environment_setup.reset()
            self.reply_memory.clear()
            # self.deep_q_environment_setup.visualize_debug()
            for step in range(self.config.correct_positioning_pre_training.batch_size):
                try:
                    state = self.deep_q_environment_setup.get_state()
                    transition = NonRLTransition(
                        state=state,
                        shape_parameters=self.deep_q_environment_setup.course_problem_setup.shape_parameters,
                        shape_transformation_parameters=self.deep_q_environment_setup.shape_transformation_parameters,
                        shaplely_polygon = self.deep_q_environment_setup.shapely_polygon,
                        expected_average_point_variance=self.config.correct_positioning_pre_training.max_reward_for_good_variance,
                        points=list(copy.deepcopy(self.deep_q_environment_setup.fine_problem_setup.p))
                    )
                    # self.deep_q_environment_setup.visualize_debug_with_input(transition.state, transition.shaplely_polygon)
                    self.reply_memory.push(transition)
                    random_step = self.parameter_server.choose_random_action()
                    random_step[1] = 1.1
                    if step > 0 and step % 7 == 0 and False:
                        self.deep_q_environment_setup.reset()
                    else:
                        self.deep_q_environment_setup.step(random_step.detach().numpy())
                except Exception as e:
                    print("simulation crashed")
                    crashed = True
                    break
            if not crashed:
                self.optimize_model()
            if ((batch_set % 3) == 0):
                self.save_models_and_state()

if __name__ == '__main__':
    position_pre_trainer = PositionPretrainer()
    position_pre_trainer.train()

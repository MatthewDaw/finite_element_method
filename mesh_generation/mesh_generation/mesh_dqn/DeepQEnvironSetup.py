
import numpy as np
from shapely.geometry import Point, Polygon
from skfem import MeshTri1

from common.exceptions import MeshBreakingError
from common.pydantic_models import ShapeTransformationParameters, ShapeOutlineParameters
from common.visuzalizers import Visualizer
from dif_eq_setup.dif_eq_setup_server import DiffEQParameterServer
from fem_solver.solver_sdk import FEMSolver
from mesh_generation.mesh_dqn.mesh_editor.mesh_editor import MeshEditor
from mesh_generation.mesh_dqn.pydantic_objects import FlowConfig
from mesh_generation.mesh_generator_sdk import MeshSDK
from mesh_generation.shape_generator import ShapeGenerator
from simulations.example_simple_simulations import FEMSimulator
from torch_geometric.data import Data
import torch
import copy
from shapely.geometry import Point, Polygon
from adaptmesh.solve import laplace
from shapely.plotting import plot_polygon

class DeepQEnvironSetup:

    def __init__(self, config: FlowConfig):
        self.config = config
        self.simulator = FEMSimulator()
        self.problem_setup = None
        self.course_problem_setup = None
        self.fine_problem_setup = None
        self.shape_transformation_parameters = None
        self.starting_num_points = None
        self.shapely_polygon = None
        self.steps = None
        self.episodes = 0
        self.error_history = []
        self.starting_error = None
        self.terminated = False
        self.NEGATIVE_REWARD = -1.*self.config.agent_params.large_neg_reward
        self.POSITIVE_REWARD = self.config.agent_params.large_neg_reward
        self.mesh_editor = MeshEditor()
        self.visuzlizer = Visualizer()
        self.reset_num = 0

    def set_shape_transformation_parameters(self, shape_outline_parameters: ShapeOutlineParameters):
        min_x = np.min(shape_outline_parameters.x_points)
        min_y = np.min(shape_outline_parameters.y_points)
        width = np.max(shape_outline_parameters.x_points) - min_x
        height = np.max(shape_outline_parameters.y_points) - min_y
        self.shape_transformation_parameters = ShapeTransformationParameters(
            x_shift=-1*min_x,
            y_shift=-1*min_y,
            x_scale=1/width,
            y_scale=1/height,
        )

    def select_random_point_in_mesh(self):
        minx, miny, maxx, maxy = self.shapely_polygon.bounds
        while True:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)
            if self.shapely_polygon.contains(point):
                return point

    def perform_scaling(self, coordinates: np.ndarray, special_transformation_parameters=None):
        """Perform scaling on the coordinates."""
        if special_transformation_parameters is None:
            special_transformation_parameters = self.shape_transformation_parameters
        coordinates[:,0] += special_transformation_parameters.x_shift
        coordinates[:,1] += special_transformation_parameters.y_shift
        coordinates[:, 0] *= special_transformation_parameters.x_scale
        coordinates[:, 1] *= special_transformation_parameters.y_scale
        return coordinates

    def undo_scaling(self, coordinates: np.ndarray, special_transformation_parameters=None):
        """Perform scaling on the coordinates."""
        if special_transformation_parameters is None:
            special_transformation_parameters = self.shape_transformation_parameters
        coordinates[:, 0] /= special_transformation_parameters.x_scale
        coordinates[:, 1] /= special_transformation_parameters.y_scale
        coordinates[:, 0] -=  special_transformation_parameters.x_shift
        coordinates[:, 1] -= special_transformation_parameters.y_shift
        return coordinates

    def reset_shape(self):
        num_triangles = np.random.randint(self.config.geom_generator_params.min_triangles,
                                          self.config.geom_generator_params.max_triangles+1)
        self.course_problem_setup = self.simulator.generate_triangle_setup(num_triangles=num_triangles,
                                                                           h=self.config.geom_generator_params.course_h)
        self.fine_problem_setup = self.simulator.generate_triangle_setup_from_outline_parameters(
            self.course_problem_setup.shape_parameters, h=self.config.geom_generator_params.fine_h)
        self.set_shape_transformation_parameters(self.course_problem_setup.shape_parameters)
        self.starting_num_points = len(self.course_problem_setup.p)
        shape_coords = list(zip(self.course_problem_setup.shape_parameters.x_points[0],
                                self.course_problem_setup.shape_parameters.y_points[0]))
        self.shapely_polygon = Polygon(shape_coords)
        self.expected_average_point_variance = self.calc_expected_average_point_variance()

    def reset(self):
        self.reset_shape()
        self.steps = 0
        self.num_episodes = 0
        self.terminated = False
        self.error_history = [self.calculate_error()]
        self.starting_error = self.error_history[0]
        self.reset_num += 1


    def get_state(self):
        # NOTE, this algorithm can be greatly optimized by using inverse t data structure
        p, t, e = self.course_problem_setup.p, self.course_problem_setup.t, self.course_problem_setup.e
        scaled_p = copy.deepcopy(p)
        scaled_p[:,:2] = self.perform_scaling(scaled_p[:, :2])

        # Create x (node features)
        x = []
        for i in range(scaled_p.shape[0]):
            is_on_edge = int(i in e[:, :2])  # Check if the point is on the edge
            # associated_triangle = t[np.where(np.any(t[:,:3] == i, axis=1))[0][0]]
            one_hot_point_type = np.zeros(10)
            # one_hot_point_type[associated_triangle[-1]] = 1
            x.append([scaled_p[i, 0], scaled_p[i, 1], is_on_edge, *one_hot_point_type])
        x = np.array(x)

        # setting one hot encoding from triangle, point is on multiple regions then it gets encoded for being in both regions
        x[t[:,0],t[:,3]+2 ] = 1
        x[t[:, 1], t[:, 3] + 2] = 1
        x[t[:, 2], t[:, 3] + 2] = 1

        # Create edge_index and edge_attr
        edge_index = []
        edge_attr = []

        def add_edge(p1, p2, edge_characteritics, is_boundary):
            """Add an edge to edge_index and edge_attr."""
            edge_index.append([p1, p2])
            distance = np.linalg.norm(self.perform_scaling(np.array([p[p1]])) - self.perform_scaling(np.array([p[p2]])))
            edge_attr.append([distance, is_boundary, *list(edge_characteritics)])

        # Add edges from t (triangle edges)
        for i in range(t.shape[0]):
            p1, p2, p3, shape_type = t[i]
            pair = np.array([p1, p2])
            # indices = np.where((e[:,:2] == pair).all(axis=1) | (e[:,:2] == pair[::-1]).all(axis=1))[0]
            extra_edge_attr = np.zeros(6)
            # if len(indices) > 0:
            #     extra_edge_attr[0] = 1
            #     extra_edge_attr[1:] = e[indices[0]][2:]
            add_edge(p1, p2, extra_edge_attr, is_boundary=0)

            pair = np.array([p2, p3])
            # indices = np.where((e[:, :2] == pair).all(axis=1) | (e[:, :2] == pair[::-1]).all(axis=1))[0]
            extra_edge_attr = np.zeros(6)
            # if len(indices) > 0:
            #     extra_edge_attr[0] = 1
            #     extra_edge_attr[1:] = e[indices[0]][2:]
            add_edge(p2, p3, extra_edge_attr, is_boundary=0)

            pair = np.array([p3, p1])
            indices = np.where((e[:, :2] == pair).all(axis=1) | (e[:, :2] == pair[::-1]).all(axis=1))[0]
            extra_edge_attr = np.zeros(6)
            if len(indices) > 0:
                extra_edge_attr[0] = 1
                extra_edge_attr[1:] = e[indices[0]][2:]
            add_edge(p3, p1, extra_edge_attr, is_boundary=0)

        edge_index = np.array(edge_index).T  # Convert to a 2 x num_edges matrix
        edge_attr = np.array(edge_attr)

        assert np.max(x[:,0]) <= 1.001
        assert np.max(x[:, 0]) >= .99
        assert np.min(x[:, 0]) == 0.0
        assert np.max(x[:,1]) <= 1.001
        assert np.max(x[:, 1]) >= .99
        assert np.min(x[:, 1]) == 0.0

        return Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index), edge_attr=torch.tensor(edge_attr)).to(self.config.device)

    def add_node(self, coordinates):
        p, t, e = self.mesh_editor.add_point(self.course_problem_setup, float(coordinates[0]), float(coordinates[1]))
        if np.isnan(p).any():
            p, t, e = self.mesh_editor.add_point(self.course_problem_setup, float(coordinates[0]),
                                                 float(coordinates[1]))
        self.course_problem_setup.p = p
        self.course_problem_setup.t = t
        self.course_problem_setup.e = e

    def remove_node(self, coordinates):
        p, t, e = self.mesh_editor.remove_point(self.course_problem_setup, float(coordinates[0]), float(coordinates[1]))
        self.course_problem_setup.p = p
        self.course_problem_setup.t = t
        self.course_problem_setup.e = e

    def calculate_error(self):
        u = self.simulator.fem_solver.solve(self.course_problem_setup.p, self.course_problem_setup.t, self.course_problem_setup.e, self.course_problem_setup.pde_coefficients)
        u = np.array(u)
        err, errorh2 = self.simulator.estimate_error_at_points(self.fine_problem_setup.p, self.fine_problem_setup.u, self.course_problem_setup.p, u, self.course_problem_setup.h)
        return err

    def calculate_reward(self):
        new_error = self.calculate_error()
        net_change_in_error = self.starting_error - new_error

        change_in_points_for_run = (self.starting_num_points - len(
            self.course_problem_setup.p)) * self.config.agent_params.expected_avg_improvement

        # if nothing is changing, give a negative reward to incentivize the agent to terminate faster
        self.error_history.append(new_error)
        if len(self.error_history) > self.config.agent_params.time_steps_to_average_improvement + 10:
            numpy_hist = np.array(self.error_history)
            improvements = numpy_hist[-self.config.agent_params.time_steps_to_average_improvement-1:-1] - numpy_hist[-self.config.agent_params.time_steps_to_average_improvement:]
            avg_improvement = np.mean(improvements)
            shifted_avg_improvement = avg_improvement - self.config.agent_params.min_expected_avg_improvement
            if shifted_avg_improvement < 0:
                self.terminated = True
                return self.NEGATIVE_REWARD * 0.5

        time_punishment = (self.steps / self.config.agent_params.max_iterations_for_episode) * self.config.agent_params.expected_avg_improvement

        return (np.exp(net_change_in_error) - 1) + change_in_points_for_run - time_punishment

    def calculate_distance_loss_for_specific_point(self, point: Point, shapely_polygon: Polygon = None):
        """Calculate the distance loss for a specific point."""
        if shapely_polygon is None:
            shapely_polygon = self.shapely_polygon
        is_contained = shapely_polygon.contains(point)
        if is_contained:
            distance_to_edge = point.distance(shapely_polygon.exterior)
            return torch.tensor(-0.4*distance_to_edge, requires_grad=True)
        else:
            distance = point.distance(shapely_polygon)
            return torch.tensor(distance+0.1, requires_grad=True)

    def run_visualization(self, points, shapely_polygons, scaling_parameters, point_to_add, point_to_remove, add_point_in_shape, point_to_add_loss, remove_point_in_shape, point_to_remove_loss):

        pionts_of_interest = np.array(
            [point_to_add.tolist()[0], point_to_remove.tolist()[0]])
        self.visualize_debug_with_input(points,
                                        shapely_polygons,
                                        scaling_parameters,
                                        pionts_of_interest
                                        )

        print(point_to_add, add_point_in_shape, point_to_add_loss)
        print(point_to_remove, remove_point_in_shape, point_to_remove_loss)

    def calc_simple_reward_for_correct_position(self, state_choice_output, scaling_parameters, shapely_polygons, points):
        """Determine if output is in the shape."""
        state_choice_output = state_choice_output.clone()
        point_to_add = self.undo_scaling(state_choice_output[4:6].reshape(1, 2), scaling_parameters)
        polygon_point = Point(point_to_add[0][0].item(), point_to_add[0][1].item())
        point_to_add_loss = self.calculate_distance_loss_for_specific_point(polygon_point, shapely_polygons)
        add_point_in_shape = point_to_add_loss <= 0
        point_to_remove = self.undo_scaling(state_choice_output[6:8].reshape(1, 2), scaling_parameters)
        polygon_point = Point(point_to_remove[0][0].item(), point_to_remove[0][1].item())
        point_to_remove_loss = self.calculate_distance_loss_for_specific_point(polygon_point, shapely_polygons)
        remove_point_in_shape = point_to_remove_loss <= 0
        # self.run_visualization(points, shapely_polygons, scaling_parameters, point_to_add, point_to_remove, add_point_in_shape, point_to_add_loss, remove_point_in_shape, point_to_remove_loss)
        return point_to_add_loss + point_to_remove_loss, add_point_in_shape, remove_point_in_shape

    def run_bulk_visualizer(self, state_choice_outputs):
        state_choice_outputs = state_choice_outputs.clone()
        self.visuzlizer.show_points_and_their_values(self.course_problem_setup.p[:, :2], points_of_interest=self.undo_scaling(state_choice_outputs[:, 4:6].detach().numpy()))
        self.visuzlizer.show_points_and_their_values(self.course_problem_setup.p[:, :2], points_of_interest=self.undo_scaling(state_choice_outputs[:, 6:8].detach().numpy()))

    def calc_expected_average_point_variance(self):
        """Calculate the max shape distribution."""

        min_x = np.min(self.course_problem_setup.shape_parameters.x_points)
        max_x = np.max(self.course_problem_setup.shape_parameters.x_points)

        min_y = np.min(self.course_problem_setup.shape_parameters.y_points)
        max_y = np.max(self.course_problem_setup.shape_parameters.y_points)

        total_points = 20 * 20

        x_breadth = max_x - min_x
        y_breadth = max_y - min_y

        x_factor = x_breadth / (x_breadth + y_breadth)
        y_factor = y_breadth / (x_breadth + y_breadth)

        estimated_area = x_factor * y_factor

        scaling_need = total_points / estimated_area

        x_factor = x_factor * np.sqrt(scaling_need)
        y_factor = y_factor * np.sqrt(scaling_need)

        x = np.linspace(min_x, max_x, int(x_factor))  # 5 points between 0 and 1 (inclusive)
        y = np.linspace(min_y, max_y, int(y_factor))
        # Create the grid
        xx, yy = np.meshgrid(x, y)
        # Stack the points
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        mask = np.array([self.shapely_polygon.contains(Point(p)) for p in grid_points])
        points_inside = grid_points[mask]
        return self.calculate_point_variance(points_inside)


    def determine_early_stop(self, state_choice_output, terminate_episode, add_point, remove_point, add_and_remove):

        if self.config.agent_params.max_iterations_for_episode <= self.steps:
            # if the algorithm hasn't terminated yet, terminate it and give it a large negative reward for not terminating itself already
            self.terminated = True
            return self.NEGATIVE_REWARD

        if terminate_episode:
            self.terminated = True
            return 0

        # if the agent is suggesting to add a point, check if the point is in the polygon
        if add_point or add_and_remove:
            add_point_in_polygon = self.shapely_polygon.contains(Point(state_choice_output[4:6]))
            reward = 0
            if not add_point_in_polygon:
                self.terminated = True
                reward = self.NEGATIVE_REWARD * 0.5
            if add_and_remove:
                remove_point_in_polygon = self.shapely_polygon.contains(Point(state_choice_output[6:8]))
                if not remove_point_in_polygon:
                    self.terminated = True
                    reward += self.NEGATIVE_REWARD * 0.5
            if self.terminated:
                return reward

        # if the agent is suggesting to remove a point, check if the point is in the polygon
        if remove_point:
            remove_point_in_polygon = self.shapely_polygon.contains(Point(state_choice_output[6:8]))
            if not remove_point_in_polygon:
                self.terminated = True
                return self.NEGATIVE_REWARD * 0.5

        # should always add point until estimate error is below threshold
        if not add_point:
            mesh = MeshTri1(self.course_problem_setup.p.T, self.course_problem_setup.t[:, :3].T)
            estimate_error = np.mean(laplace(mesh))
            if estimate_error < self.config.agent_params.min_est_error_before_removing_points:
                self.terminated = True
                return self.NEGATIVE_REWARD * 0.5

    def step(self, state_choice_output_original):
        """Running the action in the environment."""

        #     # state_choice_output dim meanings:
        #     # 0: terminate
        #     # 1: add node suggestion strength
        #     # 2: remove node suggestion strength
        #     # 3: both add and remove node suggestion strength
        #     # 4: node to add x
        #     # 5: node to add y
        #     # 6: node to remove x
        #     # 7: node to remove y

        state_choice_output = copy.deepcopy(state_choice_output_original)

        self.steps += 1

        # initialize action choices
        terminate_episode = False
        add_point = False
        remove_point = False
        add_and_remove = False

        # determine action
        max_suggestion = np.max(state_choice_output[:4])
        if state_choice_output[0] == max_suggestion:
            terminate_episode = True
        elif state_choice_output[1] == max_suggestion:
            add_point = True
        elif state_choice_output[2] == max_suggestion:
            remove_point = True
        elif state_choice_output[3] == max_suggestion:
            add_and_remove = True

        # undo scaling so coordinates match real shape
        state_choice_output[4:6] = self.undo_scaling(state_choice_output[4:6].reshape(1, 2))
        state_choice_output[6:8] = self.undo_scaling(state_choice_output[6:8].reshape(1, 2))

        # # mocks for setting up testing, I guess
        # state_choice_output[4:6] = np.array([ -0.03894506, -0.21932602])
        # state_choice_output[6:8] = np.array([ -0.03894506, -0.21932602])
        # remove_point = False
        # add_point = True
        # add_and_remove = False

        # check if the agent should stop early
        reward = self.determine_early_stop(state_choice_output, terminate_episode, add_point, remove_point, add_and_remove)
        if self.terminated:
            return reward

        try:
            if add_point:
                self.add_node(state_choice_output[4:6])
            elif remove_point:
                self.remove_node(state_choice_output[6:8])
            elif add_and_remove:
                # note, I should really have a function that does both of these at the same time
                self.remove_node(state_choice_output[6:8])
                self.add_node(state_choice_output[4:6])
        except MeshBreakingError:
            self.terminated = True
            return self.NEGATIVE_REWARD
        return self.calculate_reward()

    def calculate_point_variance(self, points):
        """Calculate the variance of the points."""
        # tree = cKDTree(points)
        # distances, _ = tree.query(points, k=2)
        # variance_distance = np.var(distances[:, 1])
        return points[:,0].var() + points[:,1].var()

    def visualize_debug_with_input(self, state, polygon, scaling_parameters, pionts_of_interest=None):
        self.visuzlizer.show_points_and_their_values(self.undo_scaling(state.x[:, :2], scaling_parameters),
                                                     shapely_polygon=polygon, points_of_interest=pionts_of_interest)

    def visualize_debug(self):
        self.visuzlizer.show_points_and_their_values(self.course_problem_setup.p[:, :2], shapely_polygon=self.shapely_polygon)



        self.visuzlizer.show_points_and_their_values(self.undo_scaling(self.perform_scaling(self.course_problem_setup.p[:, :2])),
                                                     shapely_polygon=self.shapely_polygon)

        state = self.get_state()

        self.visuzlizer.show_points_and_their_values(self.undo_scaling(state.x[:, :2]),
                                                     shapely_polygon=self.shapely_polygon)

        # zero_zero_points = np.array([[0.0, 0.0], [1.0, 1.0]])
        # transformed_points = self.undo_scaling(zero_zero_points)

        # self.visuzlizer.show_points_and_their_values(self.course_problem_setup.p[:, :2], points_of_interest=transformed_points)
        #
        # self.visuzlizer.show_points_and_their_values(self.course_problem_setup.p[:, :2],
        #                                              points_of_interest=np.array(list(self.shapely_polygon.boundary.coords)))




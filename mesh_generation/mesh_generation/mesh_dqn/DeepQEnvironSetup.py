
import numpy as np
from shapely.geometry import Point, Polygon

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
        self.last_estimated_error = None
        self.reset()
        self.do_nothing_offset = 0
        self.NEGATIVE_REWARD = -1.
        self.mesh_editor = MeshEditor(config)
        self.visuzlizer = Visualizer()



    def set_shape_transformation_parameters(self, shape_outline_parameters: ShapeOutlineParameters):
        min_x = np.min(shape_outline_parameters.x_points)
        min_y = np.min(shape_outline_parameters.y_points)
        width = np.max(shape_outline_parameters.x_points) - min_x
        height = np.max(shape_outline_parameters.y_points) - min_y
        scale = width
        if height > width:
            scale = height
        self.shape_transformation_parameters = ShapeTransformationParameters(
            x_shift=-1*min_x,
            y_shift=-1*min_y,
            scale=1/scale
        )

    def perform_scaling(self, coordinates: np.ndarray):
        """Perform scaling on the coordinates."""
        coordinates[:,0] += self.shape_transformation_parameters.x_shift
        coordinates[:,1] += self.shape_transformation_parameters.y_shift
        coordinates[:, :2] *= self.shape_transformation_parameters.scale
        return coordinates

    def undo_scaling(self, coordinates: np.ndarray):
        """Perform scaling on the coordinates."""
        coordinates[:, :2] /= self.shape_transformation_parameters.scale
        coordinates[:, 0] -=  self.shape_transformation_parameters.x_shift
        coordinates[:, 1] -= self.shape_transformation_parameters.y_shift
        return coordinates

    def reset_shape(self):
        num_triangles = np.random.randint(self.config.geom_generator_params.min_triangles,
                                          self.config.geom_generator_params.max_triangles)
        self.course_problem_setup = self.simulator.generate_triangle_setup(num_triangles=num_triangles,
                                                                           h=self.config.geom_generator_params.course_h)
        self.fine_problem_setup = self.simulator.generate_triangle_setup_from_outline_parameters(
            self.course_problem_setup.shape_parameters, h=self.config.geom_generator_params.fine_h)
        self.set_shape_transformation_parameters(self.course_problem_setup.shape_parameters)
        self.starting_num_points = len(self.course_problem_setup.p)
        shape_coords = list(zip(self.course_problem_setup.shape_parameters.x_points[0],
                                self.course_problem_setup.shape_parameters.y_points[0]))
        self.shapely_polygon = Polygon(shape_coords)

    def reset(self):
        self.reset_shape()
        self.steps = 0
        self.num_episodes = 0
        self.last_estimated_error = self.calculate_error()


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
            distance = np.linalg.norm(p[p1] - p[p2])
            edge_attr.append([distance, is_boundary, *list(edge_characteritics)])

        # Add edges from t (triangle edges)
        for i in range(t.shape[0]):
            p1, p2, p3, shape_type = t[i]
            pair = np.array([p1, p2])
            indices = np.where((e[:,:2] == pair).all(axis=1) | (e[:,:2] == pair[::-1]).all(axis=1))[0]
            extra_edge_attr = np.zeros(6)
            if len(indices) > 0:
                extra_edge_attr[0] = 1
                extra_edge_attr[1:] = e[indices[0]][2:]
            add_edge(p1, p2, extra_edge_attr, is_boundary=0)

            pair = np.array([p2, p3])
            indices = np.where((e[:, :2] == pair).all(axis=1) | (e[:, :2] == pair[::-1]).all(axis=1))[0]
            extra_edge_attr = np.zeros(6)
            if len(indices) > 0:
                extra_edge_attr[0] = 1
                extra_edge_attr[1:] = e[indices[0]][2:]
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
        return Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index), edge_attr=torch.tensor(edge_attr)).to(self.config.device)

    def add_node(self, coordinates):
        try:
            p, t, e = self.mesh_editor.add_point(self.course_problem_setup, float(coordinates[0][0]), float(coordinates[0][1]))
            self.course_problem_setup.p = p
            self.course_problem_setup.t = t
            self.course_problem_setup.e = e
        except MeshBreakingError:
            return 3
        return 4

    def remove_node(self, coordinates):
        try:
            p, t, e = self.mesh_editor.remove_point(self.course_problem_setup, float(coordinates[0][0]), float(coordinates[0][1]))
            self.course_problem_setup.p = p
            self.course_problem_setup.t = t
            self.course_problem_setup.e = e
        except MeshBreakingError:
            return 3
        return 4


    def calculate_error(self):
        u = self.simulator.fem_solver.solve(self.course_problem_setup.p, self.course_problem_setup.t, self.course_problem_setup.e, self.course_problem_setup.pde_coefficients)
        u = np.array(u)
        err, errorh2 = self.simulator.estimate_error_at_points(self.fine_problem_setup.p, self.fine_problem_setup.u, self.course_problem_setup.p, u, self.course_problem_setup.h)
        return err
        print("think more here")

    def check_if_in_shape(self, coordinates):
        """Check if the coordinates are within the shape."""
        print("think more here")

    def calculate_reward(self, new_error):
        print("think more here")

    def step(self, choice, scaled_coordinates):
        """Running the action in the environment."""
        choice = 1
        coordinates = self.undo_scaling(scaled_coordinates)
        coordinates = np.array([[ 0.83792781,  0.91771115]])

        # mesh_update_action:
        # 0: coordinates is out of bounds
        # 1: error threshold met
        # 2: max nodes reached
        # 3: mesh broken
        # 4: do nothing
        # 5: successfully removed a node
        # 6: successfully added a node

        mesh_update_action = None

        done = False

        # Do nothing
        if choice == 0:
            self.do_nothing_offset += 1
            mesh_update_action = 4

        reward = 0

        if not self.shapely_polygon.contains(Point(coordinates)):
            reward = self.NEGATIVE_REWARD
            mesh_update_action = 0
        else:
            # add a node
            if choice == 1:
                mesh_update_action = self.add_node(coordinates)
            # remove a node
            if choice == 2:
                mesh_update_action = self.remove_node(coordinates)

            # coordinate out of bounds
            if mesh_update_action == 0:
                reward = self.NEGATIVE_REWARD
            # error threshold met
            elif mesh_update_action == 1:
                reward = self.NEGATIVE_REWARD * 0.5
            # max number of nodes met
            elif mesh_update_action == 2:
                reward = self.NEGATIVE_REWARD * 0.5
            # mesh broken
            elif mesh_update_action == 3:
                reward = self.NEGATIVE_REWARD * 0.5
                done = True
            # mesh update successful
            elif mesh_update_action == 4:
                new_error = self.calculate_error()
                self.calculate_reward(new_error)

        if self.config.agent_params.max_iterations_for_episode <= self.steps:
            done = True

        if self.do_nothing_offset >= self.config.agent_params.max_do_nothing_offset:
            done = True

        if done:
            self.episodes += 1

        self.steps += 1

        return reward, done


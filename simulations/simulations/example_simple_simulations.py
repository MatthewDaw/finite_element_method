"""This file runs the two rectangle simulation given in the book."""
from matplotlib.patches import Polygon
from skfem import MeshTri1

from adaptmesh.solve import laplace
from common.pydantic_models import FullProblemSetup
from common.visuzalizers import Visualizer
from dif_eq_setup.dif_eq_setup_server import DiffEQParameterServer
from fem_solver.solver_sdk import FEMSolver
from mesh_generation.mesh_dqn.mesh_editor.mesh_editor import MeshEditor
from mesh_generation.mesh_generator_sdk import MeshSDK
from mesh_generation.shape_generator import ShapeGenerator
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import os

class FEMSimulator:
    """FEM simulator for running simulations."""

    true_answer_h = 0.02

    def __init__(self):
        self.shape_generator = ShapeGenerator()
        self.parameter_server = DiffEQParameterServer()
        self.mesh_sdk = MeshSDK()
        self.fem_solver = FEMSolver()
        self.visuzlizer = Visualizer()
        self.mesh_editor = MeshEditor()

    def validate_solution(self, p, u, exact_sol, h):
        """Validate the solution."""
        p = p.T
        x1 = p[0, :]
        x2 = p[1, :]
        u_exact = np.transpose(exact_sol(x1, x2, 1))
        u_exact = u_exact.reshape(u.shape)
        h2 = h * h
        err, errorh2 = np.linalg.norm(u_exact - u, np.inf), np.linalg.norm(u_exact - u, np.inf) / h2
        return err, errorh2

    def plot_solution(self, p, u):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(p[:,0], p[:,1], c=u, cmap='viridis', edgecolor='k')
        plt.colorbar(scatter, label='Value of u')
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.title('Scatter Plot of Points Colored by u Values')
        # plt.show()

    def run_book_disk_example_simulation(self, h=0.05, plot=False):
        disk_outline_parameters = self.shape_generator.book_disk_example()
        pde_coefficients, exact_solution = self.parameter_server.load_example_disk_setup()
        p, t, e = self.mesh_sdk.generate_adapt_mesh(disk_outline_parameters, h)
        u = self.fem_solver.solve(p, t, e, pde_coefficients)
        err, errorh2 = self.validate_solution(p, u, exact_solution, h)
        if plot:
            self.plot_solution(p, u)
        print(f"Error: {err}, Error h2: {errorh2}")

    def _load_results(self, file_name: str):
        near_exact_sol, position_of_sol_points = None, None
        folder_name = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
        file_path = os.path.join(folder_name, f"cached_solutions/near_exact_sol/{file_name}.npy")
        if os.path.exists(file_path):
            near_exact_sol = np.load(file_path)
        file_path = os.path.join(folder_name, f"cached_solutions/position_of_sol_points/{file_name}.npy")
        if os.path.exists(file_path):
            position_of_sol_points = np.load(file_path)
        return position_of_sol_points, near_exact_sol

    def _save_results(self, position_of_sol_points, near_exact_sol, file_name: str):

        position_of_sol_points_dir = "cached_solutions/position_of_sol_points"
        cached_solutions_dir = "cached_solutions/near_exact_sol"

        # Create the directories if they don't exist
        os.makedirs(position_of_sol_points_dir, exist_ok=True)
        os.makedirs(cached_solutions_dir, exist_ok=True)

        folder_name = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
        file_path = os.path.join(folder_name, f"{position_of_sol_points_dir}/{file_name}.npy")
        np.save(file_path, position_of_sol_points)
        file_path = os.path.join(folder_name, f"{cached_solutions_dir}/{file_name}.npy")
        np.save(file_path, near_exact_sol)

    def _calculate_and_save_near_exact_solution(self, shape_outline_parameters, pde_coefficients, *args, **kwargs):
        problem_hash = shape_outline_parameters.to_hash() + pde_coefficients.to_hash() + str(self.true_answer_h)
        position_of_sol_points, near_exact_sol = self._load_results(problem_hash)
        if position_of_sol_points is None or near_exact_sol is None or True:
            position_of_sol_points, t, e = self.mesh_sdk.generate_adapt_mesh(shape_outline_parameters, 0.05)
            if 'solver' in kwargs:
                near_exact_sol = kwargs['solver'](position_of_sol_points, t, e, pde_coefficients)
            else:
                near_exact_sol = self.fem_solver.solve(position_of_sol_points, t, e, pde_coefficients)
            self._save_results(position_of_sol_points, near_exact_sol, problem_hash)
        return near_exact_sol, position_of_sol_points

    def estimate_error_at_points(self, p_to_project, anchor_points_for_u, p_to_project_onto, u_to_estimate, h):
        """Estimate error at points."""
        linear_interpolator = LinearNDInterpolator(p_to_project_onto, u_to_estimate[:,0])
        u_near_exact = linear_interpolator(p_to_project)

        if np.any(np.isnan(u_near_exact)):
            nearest_interpolator = NearestNDInterpolator(p_to_project_onto, u_to_estimate[:, 0])
            nan_indices = np.isnan(u_near_exact)
            u_near_exact[nan_indices] = nearest_interpolator(p_to_project[nan_indices])

        # # we do nearest neighbor interpolation for the null evaluations
        # linear_interpolator = LinearNDInterpolator(anchor_points_for_u, anchor_points_for_u[:, 0])

        h2 = h * h
        err, errorh2 = np.linalg.norm(u_near_exact - anchor_points_for_u[:, 0]), np.linalg.norm(
            u_near_exact - anchor_points_for_u[:, 0]) / h2
        return err, errorh2
        # self.visuzlizer.show_points_and_their_values(p_to_project, u_near_exact)

    def _estimate_mesh_error(self, p, shape_outline_parameters, pde_coefficients, estimated_solution, h, *args, **kwargs):
        near_exact_sol, points_at_p = self._calculate_and_save_near_exact_solution(shape_outline_parameters, pde_coefficients, *args, **kwargs)
        linear_interpolator = LinearNDInterpolator(points_at_p, near_exact_sol[:, 0])
        u_near_exact = linear_interpolator(p)
        # Interpolate values at coarser mesh points
        h2 = h * h
        err, errorh2 = np.linalg.norm(u_near_exact - estimated_solution[:,0], np.inf), np.linalg.norm(u_near_exact - estimated_solution[:,0], np.inf) / h2
        return err, errorh2

    def generate_triangle_setup_from_outline_parameters(self, shape_parameters, h=0.05, solve=True) -> FullProblemSetup:
        boundary_labels = list(set([int(el.split(':')[0]) for el in shape_parameters.labels[0]]))
        pde_coefficients, _ = self.parameter_server.load_simple_poisson_setup(boundary_labels)
        p, t, e = self.mesh_sdk.generate_adapt_mesh(shape_parameters, h)
        # remishing does a slight transformation. I have to do this for when adding points.
        # to the delta of adding poitns more reliable, we do this here
        mesh = self.mesh_editor.remesh_with_boundaries(p, e[:,:2])
        new_p, new_t, new_e = self.mesh_editor.generate_p_t_e_from_mesh(mesh)

        u = None
        if solve:
            u = self.fem_solver.solve(new_p, new_t, new_e, pde_coefficients)
        return FullProblemSetup(
            h=h,
            shape_parameters=shape_parameters,
            pde_coefficients=pde_coefficients,
            p=p,
            t=t,
            e=e,
            u=u,
        )

    def generate_triangle_setup(self, num_triangles=1, h=0.05, solve=True):
        shape_parameters = self.shape_generator.random_triangle_polygon(num_triangles=num_triangles)
        return self.generate_triangle_setup_from_outline_parameters(shape_parameters, h, solve)

    def run_single_triangle_simulation(self, h=0.05, plot=False):
        """Run a single triangle simulation."""
        problem_setup = self.generate_triangle_setup(h, solve=True)
        if plot:
            self.plot_solution(problem_setup.p, problem_setup.u)
        err, errorh2 = self._estimate_mesh_error(problem_setup.p, problem_setup.shape_parameters, problem_setup.pde_coefficients, problem_setup.u, problem_setup.h)
        return err, errorh2

    def generate_error_drop_graph(self):
        num_triangles = np.arange(1, 20, 2)
        cum_errors = []
        cum_h2error = []
        cum_num_points = []
        cum_average_estimator = []
        h_steps = np.linspace(0.1, 0.008, 15)
        for t in num_triangles:
            shape_parameters = self.shape_generator.random_triangle_polygon(num_triangles=t)
            ground_truth_problem_setup = self.generate_triangle_setup_from_outline_parameters(shape_parameters, 0.006, True)
            errors = []
            h2error = []
            num_points = []
            average_estimator = []
            for h in h_steps:
                estimate = self.generate_triangle_setup_from_outline_parameters(shape_parameters, h, True)

                mesh = MeshTri1(estimate.p.T, estimate.t[:,:3].T)
                estimator = laplace(mesh)
                average_estimator.append(np.mean(estimator))


                err, h2_err = self.estimate_error_at_points(estimate.p, estimate.u, ground_truth_problem_setup.p, ground_truth_problem_setup.u, h)
                errors.append(err)
                h2error.append(h2_err)
                num_points.append(estimate.p.shape[0])
            cum_errors.append(errors)
            cum_h2error.append(h2error)
            cum_num_points.append(num_points)
            cum_average_estimator.append(average_estimator)
            # break
        plt.figure(figsize=(8, 6))
        for i, t in enumerate(range(len(cum_num_points))):
            plt.plot(cum_num_points[i], cum_errors[i], label=f"Error for {num_triangles[i]} triangles")
        plt.xlabel('Number of Points')
        plt.ylabel('Error')
        plt.title('Error Drop with Number of Points')
        plt.legend()
        # plt.show()
        plt.savefig('sim_graphs/error_rate_drops/error_drop.png')

        # plt.clf()
        plt.figure(figsize=(8, 6))
        for i, t in enumerate(range(len(cum_num_points))):
            plt.plot(cum_num_points[i], cum_h2error[i], label=f"H2 Error for {num_triangles[i]} triangles")
        plt.xlabel('Number of Points')
        plt.ylabel('Error')
        plt.title('H2 Error Drop with Number of Points')
        plt.legend()
        # save figure to png with matplot lib savefig

        # plt.show()
        plt.savefig('sim_graphs/error_rate_drops/h2_error_drop.png')

        plt.figure(figsize=(8, 6))
        # plot average estimator
        for i, t in enumerate(range(len(cum_num_points))):
            plt.plot(cum_num_points[i], cum_average_estimator[i], label=f"Average Estimator for {num_triangles[i]} triangles")
        plt.xlabel('Number of Points')
        plt.ylabel('Average Estimator')
        plt.title('h with Average Estimator')
        plt.legend()
        # plt.show()
        # save figure to png with matplot lib savefig
        plt.savefig('sim_graphs/error_rate_drops/h_to_average_estimator.png')

        # plot number of points
        # plt.clf()
        plt.figure(figsize=(8, 6))
        for i, t in enumerate(range(len(cum_num_points))):
            plt.plot(h_steps, cum_num_points[i], label=f"Number of Points for {num_triangles[i]} triangles")
        plt.xlabel('h')
        plt.ylabel('Number of Points')
        plt.title('h with Number of Points')
        plt.legend()
        # plt.show()
        # save figure to png with matplot lib savefig
        plt.savefig('sim_graphs/error_rate_drops/h_to_num_points.png')

        print("think more here")


# simulations/simulations/sim_graphs/error_rate_drops

if __name__ == '__main__':
    fem_simulator = FEMSimulator()
    fem_simulator.generate_error_drop_graph()
    # fem_simulator.run_single_triangle_simulation(0.2, plot=True)




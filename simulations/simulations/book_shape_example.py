"""This file runs the two rectangle simulation given in the book."""
from dif_eq_setup.dif_eq_setup_server import DiffEQParameterServer
from fem_solver.solver_sdk import FEMSolver
from mesh_generation.mesh_generator_sdk import MeshSDK
from mesh_generation.shape_generator import ShapeGenerator

class FEMSimulator:

    def __init__(self):
        self.shape_generator = ShapeGenerator()
        self.parameter_server = DiffEQParameterServer()
        self.mesh_sdk = MeshSDK()
        self.fem_solver = FEMSolver()

    def run_book_disk_example_simulation(self):
        disk_outline_parameters = self.shape_generator.book_disk_example()
        parameter_server = DiffEQParameterServer()
        pde_coefficients, exact_solution = self.parameter_server.load_example_disk_setup()
        p, t, e = self.mesh_sdk.generate_geom_mesh(disk_outline_parameters, 0.05)
        print("think more here")

    def run_single_triangle_simulation(self):
        triangle_outline_parameters = self.shape_generator.random_triangle_polygon()
        boundary_labels = list(set([int(el.split(':')[0]) for el in triangle_outline_parameters.labels[0]]))
        pde_coefficients, _ = self.parameter_server.load_simple_poisson_setup(boundary_labels)
        p, t, e = self.mesh_sdk.generate_adapt_mesh(triangle_outline_parameters, 0.05)
        u = self.fem_solver.solve(p, t, e, pde_coefficients)
        return u

if __name__ == '__main__':
    fem_simulator = FEMSimulator()
    fem_simulator.run_book_disk_example_simulation()

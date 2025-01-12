"""Module for adding a new point to a mesh."""
import copy

from common.exceptions import MeshBreakingError
from common.pydantic_models import FullProblemSetup
import numpy as np
import matplotlib.pyplot as plt
from common.visuzalizers import Visualizer
from scipy.spatial import Delaunay

from skfem import MeshTri1

# m = MeshTri1(np.array(p).T, np.array(t).T)
# np.array(p).T.shape
# (2, 7)
# np.array(t).T.shape
# (3, 5)

from adaptmesh.smooth import cpt

from mesh_generation.mesh_dqn.pydantic_objects import FlowConfig


class MeshEditor:

    def __init__(self, config: FlowConfig):
        self.visuzlizer = Visualizer()
        self.config = config

    def remesh(self, p):
        tri = Delaunay(p)
        try:
            cells = tri.simplices
        except ValueError:
            raise MeshBreakingError()
        mesh = MeshTri1(p.T, cells.T)
        # self.visuzlizer.show_points_and_their_values(mesh.p.T)
        # self.visuzlizer.show_points_and_their_values(mesh.p.T, points_of_interest=np.array([cow]))
        try:
            smoothed_mesh = cpt(mesh, smooth_steps=3)
        except Exception as err:
            print(err)
        return smoothed_mesh

    def get_edges_to_tiangles(self, triangles, edges):
        # Step 1: Extract edges from triangles
        # Normalize edges to always have (min, max) order for undirected comparison
        normalized_edges = [tuple(sorted(edge)) for edge in edges]
        edge_to_index = {edge: idx for idx, edge in enumerate(normalized_edges)}

        # Step 1: Extract edges from triangles and find matches
        triangle_to_edge_map = []  # Stores (edge_index, triangle_index)

        for tri_idx, triangle in enumerate(triangles):
            # Get all edges of the triangle
            triangle_edges = [
                tuple(sorted((triangle[0], triangle[1]))),
                tuple(sorted((triangle[1], triangle[2]))),
                tuple(sorted((triangle[2], triangle[0]))),
            ]
            # Check if each edge is in the edge list
            for edge in triangle_edges:
                if edge in edge_to_index:
                    triangle_to_edge_map.append((edge_to_index[edge], tri_idx))

        return triangle_to_edge_map

    def add_point(self, problem_setup: FullProblemSetup, x: float, y: float):
        """Add a point to the mesh."""

        original_p_positions = problem_setup.p[problem_setup.e[:, 0]]

        # self.visuzlizer.show_points_and_their_values(problem_setup.p)
        new_point = np.array([[x, y]])
        p_with_new_point = np.vstack([problem_setup.p, new_point])
        smoothed_mesh = self.remesh(p_with_new_point)

        p = smoothed_mesh.p.T
        t = smoothed_mesh.t.T
        # note, really we should be correcting the new dimensions of t here, but they are always zero so fixing this will have to revisited
        t = np.hstack((t, np.zeros((t.shape[0], 1))))
        e = problem_setup.e
        p[e[:, 0]] = original_p_positions

        new_point_index = np.max(t)
        new_boundary_nodes = smoothed_mesh.boundary_nodes()
        if int(new_point_index) in new_boundary_nodes:
            center_points_of_edges = (p[e[:, 0]] + p[e[:, 1]]) / 2

            self.visuzlizer.show_points_and_their_values(center_points_of_edges)

            distances = center_points_of_edges - p[int(new_point_index)]
            distances = np.linalg.norm(distances, axis=1)
            closest_edge_index = np.argmin(distances)
            closest_edge = e[closest_edge_index]
            e = np.delete(e, closest_edge_index, axis=0)
            new_entries = np.array([[closest_edge[0], new_point_index] + list(closest_edge[2:]),
            [closest_edge[1], new_point_index] + list(closest_edge[2:])])
            e = np.vstack([e, new_entries])
        return p, t.astype(int), e.astype(int)



    def remove_point(self, problem_setup: FullProblemSetup, x: float, y: float):
        """Remove a point from the mesh."""

        # save original p positions at border so that we can move them back there once we are done with our delete
        original_p_positions = problem_setup.p[problem_setup.e[:,0]]

        # find point closest to the x and y target the ML model chose
        target_point = np.array([x, y])
        distances = problem_setup.p - target_point
        normed_distances = np.linalg.norm(distances, axis=1)
        # guarantee that a boundary point is not removed
        normed_distances[problem_setup.e[:,0]] = np.inf
        # chosen point is determined
        closest_row_index = np.argmin(normed_distances)

        # delete chosen point from p
        p_without_closest = np.delete(problem_setup.p, closest_row_index, axis=0)
        smoothed_mesh = self.remesh(p_without_closest)

        return self.generate_p_t_e_from_mesh(smoothed_mesh)

        # # remesh does not resort points, so we can use p and t without any changes, other than adding labels of t back in
        # p = smoothed_mesh.p.T
        # t = smoothed_mesh.t.T
        # # note, really we should be correcting the new dimensions of t here, but they are always zero so fixing this will have to revisited
        # t = np.hstack((t, np.zeros((t.shape[0], 1), dtype=t.dtype)))
        #
        # self.visuzlizer.show_points_and_their_values(p[np.where(normed_distances < np.inf)])
        #
        # assert closest_row_index not in problem_setup.e[:,0] and closest_row_index not in problem_setup.e[:,1]
        # e = problem_setup.e
        # e[np.where(closest_row_index < problem_setup.e[:, 0]), 0] -= 1
        # e[np.where(closest_row_index < problem_setup.e[:, 1]), 1] -= 1
        #
        # self.visuzlizer.show_points_and_their_values(p[e[:, 0]])
        #
        # # reset boundary point positions, we do not want to move those as they might be drawn to the interior
        # p[e[:, 0]] = original_p_positions
        #
        # return p, t.astype(int), e.astype(int)


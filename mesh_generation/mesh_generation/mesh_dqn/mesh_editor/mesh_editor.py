"""Module for adding a new point to a mesh."""
import copy

from common.exceptions import MeshBreakingError
from common.pydantic_models import FullProblemSetup
import numpy as np
import matplotlib.pyplot as plt
from common.visuzalizers import Visualizer
from scipy.spatial import Delaunay

from skfem import MeshTri1
import triangle

# m = MeshTri1(np.array(p).T, np.array(t).T)
# np.array(p).T.shape
# (2, 7)
# np.array(t).T.shape
# (3, 5)

from adaptmesh.smooth import cpt

from mesh_generation.mesh_dqn.pydantic_objects import FlowConfig


class MeshEditor:

    def __init__(self):
        self.visuzlizer = Visualizer()

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

    def remesh_with_boundaries(self, p, boundary_edges):
        """
        Generate a constrained Delaunay triangulation, preserving boundary edges.

        Parameters:
            p (np.ndarray): Nx2 array of points (vertices).
            boundary_edges (np.ndarray): Mx2 array of indices representing boundary edges.

        Returns:
            dict: A dictionary containing 'vertices' and 'triangles' for the mesh.
        """
        # Prepare input for the `triangle` library
        points = p
        segments = boundary_edges
        # Define the input dictionary for `triangle`
        tri_input = {
            "vertices": points,
            "segments": segments,
        }
        # Perform constrained triangulation
        tri_output = triangle.triangulate(tri_input, "p")  # "p" for constrained triangulation, "q" for quality mesh
        mesh = MeshTri1(p.T, tri_output['triangles'].T)
        smoothed_mesh = cpt(mesh, smooth_steps=3)
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

    def generate_p_t_e_from_mesh(self, smoothed_mesh):
        p = smoothed_mesh.p.T
        t = smoothed_mesh.t.T
        t = np.hstack((t, np.zeros((t.shape[0], 1), dtype=t.dtype)))
        boundary_facets = smoothed_mesh.boundary_facets()
        boundary_edges = smoothed_mesh.facets[:, boundary_facets]
        boundary_nodes = smoothed_mesh.boundary_nodes()
        e = np.zeros((len(boundary_nodes), 7))
        e[:,:2] = boundary_edges.T
        e[:, 4] = 1
        e[:, 5] = 1
        return p, t.astype(int), e.astype(int)

    def add_point_and_update_edges(self, p, e, p_new, tolerance=1e-2, point_tolerance=1e-6):
        """
        Adds a new point to the mesh and adjusts the edges.
        If the new point is close to an edge, delete that edge and create two new edges connecting the new point.

        Args:
            p (np.ndarray): Array of existing points (N x 2).
            e (np.ndarray): Array of edges (M x 2).
            p_new (np.ndarray): New point to be added (1 x 2).
            tolerance (float): Distance tolerance to determine if the point is "close" to an edge.

        Returns:
            p_updated (np.ndarray): Updated points array.
            e_updated (np.ndarray): Updated edges array.
        """

        def point_to_line_projection(p1, p2, p_new):
            """Project p_new onto the line segment (p1, p2) and compute distance."""
            p1 = np.array(p1)
            p2 = np.array(p2)
            v = p2 - p1
            w = p_new - p1
            proj_scalar = np.clip(np.dot(w, v) / np.dot(v, v), 0, 1)
            proj_point = p1 + proj_scalar * v
            distance = np.linalg.norm(proj_point - p_new)
            return proj_point, distance

        distances = np.linalg.norm(p - p_new, axis=1)
        if np.min(distances) <= point_tolerance:
            # If p_new is too close to an existing point, do nothing
            return p, e, False

        closest_edge = None
        closest_proj = None
        min_distance = float('inf')

        # Check all edges for proximity
        for i, edge in enumerate(e):
            p1, p2 = p[edge[0]], p[edge[1]]
            proj_point, distance = point_to_line_projection(p1, p2, p_new)

            if distance < min_distance and distance <= tolerance:
                min_distance = distance
                closest_edge = edge
                closest_proj = proj_point

        # If no close edge is found, add the point without modifying edges
        if closest_edge is None:
            p_updated = np.vstack([p, p_new])
            e_updated = e
            return p_updated, e_updated, True

        # Snap the point to the edge
        snapped_point = closest_proj

        # Add the snapped point to the points array
        p_updated = np.vstack([p, snapped_point])
        new_point_index = len(p)

        # Remove the closest edge and add two new edges
        edge_start, edge_end = closest_edge
        e_updated = e[~np.all(e == closest_edge, axis=1)]  # Remove the closest edge
        new_edges = np.array([[edge_start, new_point_index], [new_point_index, edge_end]])
        e_updated = np.vstack([e_updated, new_edges])

        return p_updated, e_updated, True

    def add_point(self, problem_setup: FullProblemSetup, x: float, y: float):
        """Add a point to the mesh."""
        new_point = np.array([[x, y]])
        p_updated, e_updated, updated = self.add_point_and_update_edges(problem_setup.p, problem_setup.e[:,:2], new_point)
        if updated:
            smoothed_mesh = self.remesh_with_boundaries(p_updated, e_updated)
            return self.generate_p_t_e_from_mesh(smoothed_mesh)
        return problem_setup.p, problem_setup.t, problem_setup.e


    def remove_point_and_update_edges(self, p, e, i):
        """
        Removes a point at index i from p and updates the edges in e.
        If the point is part of any edges, the touching edges are deleted,
        and a new edge is added between the remaining endpoints of those edges.

        Args:
            p (np.ndarray): Array of points (N x 2).
            e (np.ndarray): Array of edges (M x 2).
            i (int): Index of the point to remove.

        Returns:
            p_new (np.ndarray): Updated array of points.
            e_new (np.ndarray): Updated array of edges.
        """
        # Find edges that include the point i
        edges_to_remove = e[(e[:, 0] == i) | (e[:, 1] == i)]

        # Get the unique endpoints of the edges to remove
        endpoints = np.unique(edges_to_remove.flatten())

        # Remove the point i from endpoints
        endpoints = endpoints[endpoints != i]

        # If there are exactly two remaining endpoints, connect them
        if len(endpoints) == 2:
            new_edge = np.array([endpoints])
        else:
            new_edge = np.empty((0, 2), dtype=int)  # No new edge needed

        # Remove edges containing point i
        e_new = e[(e[:, 0] != i) & (e[:, 1] != i)]

        # Add the new edge, if applicable
        if new_edge.size > 0:
            e_new = np.vstack([e_new, new_edge])

        # Remove the point from p
        p_new = np.delete(p, i, axis=0)

        # Adjust edge indices to account for the removed point
        e_new[e_new > i] -= 1

        return p_new, e_new

    def remove_point(self, problem_setup: FullProblemSetup, x: float, y: float):
        """Remove a point from the mesh."""
        # find point closest to the x and y target the ML model chose
        target_point = np.array([x, y])
        distances = problem_setup.p - target_point
        normed_distances = np.linalg.norm(distances, axis=1)
        closest_row_index = np.argmin(normed_distances)
        new_p, new_e = self.remove_point_and_update_edges(problem_setup.p, problem_setup.e[:,:2], closest_row_index)
        smoothed_mesh = self.remesh_with_boundaries(new_p, new_e)
        return self.generate_p_t_e_from_mesh(smoothed_mesh)

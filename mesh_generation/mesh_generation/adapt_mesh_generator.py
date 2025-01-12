"""Adapt mesh generator module."""

from common.pydantic_models import ShapeOutlineParameters
from adaptmesh import triangulate
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

class Triangulation(object):
    """Triangulation data structure"""

    # This represents the mesh
    def __init__(self):
        self.vertices = []
        self.triangles = []
        self.external = (
            None  # infinite, external triangle (outside convex hull)
        )

class AdaptMesher:

    def find_closest_segments_vectorized(self, coarse_points, fine_points):
        """
        Maps finer mesh points to line segments in the coarser mesh using vectorized operations.

        Args:
            coarse_points (np.ndarray): Array of shape (M, 2) representing coarse boundary points.
            fine_points (np.ndarray): Array of shape (N, 2) representing finer boundary points.

        Returns:
            np.ndarray: Array of shape (N,) containing the index of the coarse segment for each fine point.
        """
        # Define line segments in the coarse mesh
        segment_starts = coarse_points[:-1]  # Start points of segments
        segment_ends = coarse_points[1:]  # End points of segments
        segment_vectors = segment_ends - segment_starts  # Segment vectors

        # Compute the projection of each fine point onto each segment
        segment_vectors_norm = np.sum(segment_vectors ** 2, axis=1)  # |segment_vectors|^2

        # Reshape for broadcasting
        fine_points_expanded = fine_points[:, np.newaxis, :]  # Shape (N, 1, 2)
        segment_starts_expanded = segment_starts[np.newaxis, :, :]  # Shape (1, M-1, 2)
        segment_vectors_expanded = segment_vectors[np.newaxis, :, :]  # Shape (1, M-1, 2)

        # Vectors from segment starts to fine points
        point_vectors = fine_points_expanded - segment_starts_expanded  # Shape (N, M-1, 2)

        # Projection scalar t for each point onto each segment
        t = np.sum(point_vectors * segment_vectors_expanded, axis=2) / segment_vectors_norm
        t = np.clip(t, 0, 1)  # Clip to ensure projections are within the segment

        # Compute projection points
        projections = segment_starts_expanded + t[:, :, np.newaxis] * segment_vectors_expanded  # Shape (N, M-1, 2)

        # Compute distances from fine points to their projections
        distances = np.linalg.norm(fine_points_expanded - projections, axis=2)  # Shape (N, M-1)

        # Find the closest segment for each fine point
        closest_segments = np.argmin(distances, axis=1)  # Shape (N,)

        return closest_segments

    def is_point_on_line(self, p, line_start, line_end, tolerance=.1):
        x1, y1 = line_start
        x2, y2 = line_end
        x, y = p

        # Handle vertical line case (infinite slope)
        if abs(x2 - x1) < tolerance:
            return abs(x - x1), min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

        # Calculate slope (m)
        m = (y2 - y1) / (x2 - x1)

        # Calculate the y-value for the given x based on the line equation
        y_line = m * (x - x1) + y1

        # Check if the point lies on the line and is within the bounds of the line segment
        return abs(y - y_line), min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

    def fit_points_to_line_segment(self, coarse_points, fine_points):
        # Determine which line each point belongs to
        results = []
        left_points = coarse_points
        right_points = np.array(list(coarse_points[1:]) + [coarse_points[0]])
        for point in fine_points:
            min_error = np.inf
            min_error_index = 0
            for i, (left, right) in enumerate(zip(left_points, right_points)):
                error, is_within_bounds = self.is_point_on_line(point, left, right)
                if error < min_error and is_within_bounds:
                    min_error = error
                    min_error_index = i
            results.append(min_error_index)
        return np.array(results)

    def plot_segments_and_points(self, coarse_points, fine_points, mapping):
        """
        Plots the coarse mesh segments and fine points, coloring the fine points based on their associated segment.

        Args:
            coarse_points (np.ndarray): Array of shape (M, 2) representing coarse boundary points.
            fine_points (np.ndarray): Array of shape (N, 2) representing finer boundary points.
            mapping (np.ndarray): Array of shape (N,) mapping each fine point to a coarse segment index.
        """
        # Plot coarse segments
        colors = plt.cm.rainbow(np.linspace(0, 1, len(coarse_points) - 1))

        # Plot coarse segments with their colors
        for i in range(len(coarse_points) - 1):
            start, end = coarse_points[i], coarse_points[i + 1]
            plt.plot([start[0], end[0]], [start[1], end[1]], color=colors[i], label=f'Segment {i}', lw=2)

        # Plot fine points with the same colors as their associated segments
        for i, point in enumerate(fine_points):
            segment = mapping[i]
            plt.scatter(point[0], point[1], color=colors[segment])

        plt.title("Coarse Segments and Fine Points with Matching Colors")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.legend()
        plt.show()

    def plot_points_only(self, fine_points):
        """
        Plots the finer mesh points, coloring them based on their associated coarse segment.

        Args:
            fine_points (np.ndarray): Array of shape (N, 2) representing finer boundary points.
            mapping (np.ndarray): Array of shape (N,) mapping each fine point to a coarse segment index.
            coarse_segments_count (int): Number of coarse segments.
        """
        # Generate colors for each segment
        plt.scatter(fine_points[:, 0], fine_points[:, 1], color='blue', label='Fine Points')
        plt.title("Fine Points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.legend()
        plt.show()

    def _angle_from_reference(self, p1, p2):
        # Calculate the angle between the line formed by p1 and p2 with respect to the x-axis
        return np.linalg.norm(p2 - p1)

    def _order_points(self, points):
        # Step 1: Choose a reference point (typically the point with the smallest y, then x)
        reference_point = min(points, key=lambda p: (p[1], p[0]))

        # Step 2: Sort the remaining points by angle relative to the reference point
        sorted_points = sorted(enumerate(points), key=lambda p: self._angle_from_reference(reference_point, p[1]))

        # Step 3: Return the sorted points including the reference point first
        sorted_indices = [index for index, _ in sorted_points]
        return sorted_indices

    def _get_edges_from_ordered_points(self, points):
        sorted_indices = self._order_points(points)

        # Step 4: Create edges (connect each point to the next one using indices)
        edges = []
        for i in range(len(sorted_indices)):
            edge = (sorted_indices[i], sorted_indices[(i + 1) % len(sorted_indices)])
            edges.append(edge)

        return edges

    def run_mesh(self, shape_outline_parameters: ShapeOutlineParameters, h: float):
        points = [(x, y) for x, y in zip(shape_outline_parameters.x_points[0], shape_outline_parameters.y_points[0])]
        points_reserved = points.copy()
        if points[0] == points[-1]:
            m = triangulate(points[:-1], quality=1-h)
        else:
            m = triangulate(points, quality=1-h)
        # Extract the matrices
        p_skfem = m.p  # Node coordinates
        t_skfem = m.t  # Element connectivity

        p = p_skfem.T
        t = np.zeros((t_skfem.shape[1], 4))

        t[:, 0:3] = t_skfem.T

        boundary_facets = m.boundary_facets()

        # Map these facets to their corresponding point indices
        boundary_edges = m.facets[:, boundary_facets]
        boundary_nodes = m.boundary_nodes()
        e = np.zeros((len(boundary_nodes), 7))
        e[:,:2] = boundary_edges.T
        e[:, 5] = 1
        coarse_points, fine_points = np.array(points_reserved), m.p.T[m.boundary_nodes()]
        mapping = self.fit_points_to_line_segment(coarse_points, fine_points)

        # TODO: this label assignment is not correct, not needed for deep learning but will need to revisit and fix later
        for edge_index, label_idx in enumerate(mapping):
            label = shape_outline_parameters.labels[0][label_idx]
            split_label = label.split(":")
            e[edge_index, 4] = int(split_label[0])
            e[edge_index, 5] = int(split_label[1][1:])
            e[edge_index, 6] = int(split_label[2][1:])

        return p, t.astype(int), e.astype(int)

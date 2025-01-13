
import matplotlib.pyplot as plt
import random
import math
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
import numpy as np
from common.pydantic_models import ShapeOutlineParameters

class ShapeGenerator:
    """Class to generate shapely shapes."""

    def __init__(self, shape_size_h=0.5):
        self.shape_size_h = shape_size_h

    def visualize_plotly_merged_shape(self, merged_shape):
        """Visualize the merged shape using Plotly."""
        fig, ax = plt.subplots()
        if isinstance(merged_shape, MultiPolygon):
            for poly in merged_shape.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='blue', ec='black')
        else:
            x, y = merged_shape.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='blue', ec='black')

        plt.show()

    def _angle_between_points(self, A, B, C):
        """Calculate the angle between three points."""
        # Vectors AB and BC
        AB = (B[0] - A[0], B[1] - A[1])
        BC = (C[0] - B[0], C[1] - B[1])

        # Dot product and magnitudes of the vectors
        dot_product = AB[0] * BC[0] + AB[1] * BC[1]
        magnitude_AB = math.sqrt(AB[0] ** 2 + AB[1] ** 2)
        magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)

        # Calculate the cosine of the angle using the dot product formula
        cos_angle = dot_product / (magnitude_AB * magnitude_BC)

        # Ensure the value is within the valid range for acos due to floating-point precision
        cos_angle = max(-1.0, min(1.0, cos_angle))

        # Calculate the angle in radians, then convert to degrees
        angle = math.acos(cos_angle)
        return math.degrees(angle)

    # Function to check if all angles in the triangle are >= 30 degrees
    def _is_valid_triangle(self, triangle, min_angle=30):
        """Check if all angles in a triangle are greater than a specified minimum angle."""
        A, B, C = triangle
        angles = [
            self._angle_between_points(A, B, C),  # Angle at B
            self._angle_between_points(B, A, C),  # Angle at A
            self._angle_between_points(C, A, B)  # Angle at C
        ]
        return all(angle >= min_angle for angle in angles)

    def _triangle_area(self, triangle):
        """Calculate the area of a triangle using the shoelace formula."""
        A, B, C = triangle
        return 0.5 * abs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))

    def _generate_triangle_at_center(self, center):
        """Generate a random triangle with a specified center."""
        # Extract the center coordinates
        x_center, y_center = center

        # Set a minimum angle for valid triangles (for example, between 20 and 60 degrees)
        min_angle = np.random.uniform(20, 58)

        while True:
            # Generate three points for the triangle with the given center
            point1 = (self.shape_size_h * random.uniform(-1, 1), self.shape_size_h * random.uniform(-1, 1))
            point2 = (self.shape_size_h * random.uniform(-1, 1), self.shape_size_h * random.uniform(-1, 1))
            point3 = (self.shape_size_h * random.uniform(-1, 1), self.shape_size_h * random.uniform(-1, 1))

            # Calculate the current centroid of the generated triangle
            current_centroid = (
                (point1[0] + point2[0] + point3[0]) / 3,
                (point1[1] + point2[1] + point3[1]) / 3
            )

            # Calculate the shift required to move the current centroid to the desired center
            shift_x = x_center - current_centroid[0]
            shift_y = y_center - current_centroid[1]

            # Shift all points to move the centroid to the desired center
            point1 = (point1[0] + shift_x, point1[1] + shift_y)
            point2 = (point2[0] + shift_x, point2[1] + shift_y)
            point3 = (point3[0] + shift_x, point3[1] + shift_y)

            # Check if the triangle has valid angles
            if self._is_valid_triangle([point1, point2, point3], min_angle=min_angle):

                # Calculate the area of the triangle
                area = self._triangle_area([point1, point2, point3])

                # Scale the triangle to have area = 1 (if the area is non-zero)
                if area != 0:
                    scaling_factor = np.sqrt(1 / area)
                else:
                    scaling_factor = 1

                # Scale each point by the scaling factor relative to the centroid
                scaled_triangle = [
                    (point1[0] * scaling_factor, point1[1] * scaling_factor),
                    (point2[0] * scaling_factor, point2[1] * scaling_factor),
                    (point3[0] * scaling_factor, point3[1] * scaling_factor),
                ]

                return scaled_triangle

    def _add_random_translation(self, triangle):
        """Add a random translation between 0 and 1 in a random direction."""
        # Generate a random direction (angle) and magnitude for the translation
        angle = random.uniform(0, 2 * np.pi)  # Random angle in radians
        magnitude = random.uniform(0, 1)  # Random translation magnitude between 0 and 1

        # Calculate the translation vector
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)

        # Translate the points of the triangle
        translated_triangle = [
            (x + dx, y + dy) for x, y in triangle
        ]
        return translated_triangle

    def _generate_triangle_distribution(self, n, reduction_rate=2):
        remainder = n
        sizes = []
        while math.floor(remainder) > 0:
            sizes.append(int(math.ceil(remainder / reduction_rate))+1)
            remainder = remainder / reduction_rate
        return sizes

    def random_triangle_polygon(self, num_triangles=1) -> ShapeOutlineParameters:
        """Generate a random triangle contour."""
        sizes = self._generate_triangle_distribution(num_triangles)
        group_count = 0
        group_in_count = sizes[0]
        triangle_center = (0, 0)
        triangles = []
        for t_count in range(num_triangles):
            chosen_triangle = self._generate_triangle_at_center(triangle_center)
            triangles.append(self._add_random_translation(chosen_triangle))
            if group_in_count == 0:
                group_count += 1
                if group_count < len(sizes):
                    group_in_count = sizes[group_count]
                else:
                    group_in_count = 3
                triangle_center = (triangle_center[0]+np.random.uniform(-1, 1), triangle_center[1]+np.random.uniform(-1, 1))
            group_in_count -= 1

        # Convert triangles to polygons and merge them
        polygons = [Polygon(triangle) for triangle in triangles]
        merged_shape = unary_union(polygons)

        if isinstance(merged_shape, MultiPolygon):
            selected_x, selected_y = None, None
            for poly in merged_shape.geoms:
                x, y = poly.exterior.xy
                if selected_x is None or len(x) > len(selected_x):
                    selected_x, selected_y = x, y
        else:
            selected_x, selected_y = merged_shape.exterior.xy

        if selected_x[0] == selected_x[-1] and selected_y[0] == selected_y[-1]:
            selected_x = selected_x[:-1]
            selected_y = selected_y[:-1]

        x_points = [list(selected_x)]
        y_points = [list(selected_y)]
        z_points = [[0 for _ in range(len(selected_x))]]

        fake_arc_center = [0.0, 0.0, 0.0]
        rect1_arc_center = [[fake_arc_center for _ in range(len(selected_x))]]
        edge_types = [['line' for _ in range(len(selected_x))]]
        labels = [[f"1:L1:R0" for i in range(len(selected_x))]]
        line_loop_groups = [[0]]
        return ShapeOutlineParameters(x_points=x_points, y_points=y_points, z_points=z_points, arc_center=rect1_arc_center,
                                      edge_types=edge_types, labels=labels, line_loop_groups=line_loop_groups)

    def book_example_shape(self) -> ShapeOutlineParameters:
        """Generate the example shape from the book. This shape is two rectangles stacked on top of each other. The top has a rectangle inside a circle, and the bottom has a circle hole."""
        fake_arc_center = [0.0,0.0,0.0]
        rect1_x = [0.0,0.0,2.0,2.0]
        rect1_y = [-1.0,0.0,0.0,-1.0]
        rect1_z = [0,0,0,0]
        rect1_arc_center = [fake_arc_center,fake_arc_center,fake_arc_center,fake_arc_center]
        rect1_labels = [":L0:R1",":L3:R1",":L0:R1",":L0:R1"]

        circle_center = [1.0,-0.5,0.0]
        circle_x = [0.75,1.0,1.25,1.0]
        circle_y = [-0.5,-0.75,-0.5,-0.25]
        circle_z = [0,0,0,0]
        circle_arc_center = [circle_center,circle_center,circle_center,circle_center]
        circle_labels = [":L0:R1",":L0:R1",":L0:R1",":L0:R1"]


        rect2_x = [0.75,0.75,1.25,1.25]
        rect2_y = [0.25,0.75,0.75,0.25]
        rect2_z = [0,0,0,0]
        rect2_arc_center = [fake_arc_center,fake_arc_center,fake_arc_center,fake_arc_center]
        rect2_labels = [":L3:R2", ":L3:R2", ":L3:R2", ":L3:R2"]

        rect3_x = [0.0,0.0,2.0,2.0]
        rect3_y = [0.0,1.0,1.0,0.0]
        rect3_z = [0,0,0,0]
        rect3_arc_center = [fake_arc_center,fake_arc_center,fake_arc_center,fake_arc_center]
        rect3_labels = [":L0:R3", ":L0:R3", ":L0:R3", ":L0:R3"]

        x_points = [rect1_x,circle_x,rect2_x,rect3_x]
        y_points = [rect1_y,circle_y,rect2_y,rect3_y]
        z_points = [rect1_z,circle_z,rect2_z,rect3_z]
        arc_center = [rect1_arc_center,circle_arc_center,rect2_arc_center,rect3_arc_center]
        edge_types = [['line','line','line','line'],['arc','arc','arc','arc'],['line','line','line','line'],['line','line','line','line']]
        labels = [rect1_labels,circle_labels,rect2_labels,rect3_labels]

        label_count = 1
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                labels[i][j] = str(label_count)+labels[i][j]
                label_count += 1
        line_loop_groups = [[0,1],[2],[3,2]]
        return ShapeOutlineParameters(x_points=x_points, y_points=y_points, z_points=z_points, arc_center=arc_center,
                               edge_types=edge_types, labels=labels, line_loop_groups=line_loop_groups)

    def book_disk_example(self) -> ShapeOutlineParameters:
        """Generate the disk example from the book."""
        circle_center = [0.0, 0.0, 0.0]
        circle_x = [-1, 0, 1, 0]
        circle_y = [0,-1,0,1]
        circle_z = [0,0,0,0]
        circle_center = [circle_center,circle_center,circle_center,circle_center]
        circle_labels = ["4:L1:R0", "1:L1:R0", "2:L1:R0", "3:L1:R0"]
        x_points = [circle_x]
        y_points = [circle_y]
        z_points = [circle_z]
        arc_center = [circle_center]
        edge_types = [['arc','arc','arc','arc']]
        labels = [circle_labels]
        line_loop_groups = [[0]]
        return ShapeOutlineParameters(x_points=x_points, y_points=y_points, z_points=z_points, arc_center=arc_center, edge_types=edge_types, labels=labels, line_loop_groups=line_loop_groups)

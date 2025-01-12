"""Main software dev kit for managing mesh generation functions."""
from common.pydantic_models import ShapeOutlineParameters
from mesh_generation.shape_generator import ShapeGenerator
import numpy as np
import copy
import pygmsh
# from pygmsh import Point
import gmsh
import matplotlib.pyplot as plt

class GeomMeshGenerator:
    """Generate a mesh using pygmsh."""

    def _validate_line_points(self, shape_x_points, shape_y_points, shape_z_points):
        if shape_x_points[0] == shape_x_points[-1] and shape_y_points[0] == shape_y_points[-1] and shape_z_points[0] == \
                shape_z_points[-1]:
            shape_x_points = shape_x_points[:-1]
            shape_y_points = shape_y_points[:-1]
            shape_z_points = shape_z_points[:-1]
        if len(shape_x_points) != len(shape_y_points) or len(shape_x_points) != len(shape_z_points) or len(
                shape_y_points) != len(shape_z_points):
            raise ValueError("The number of x, y, and z points must be the same.")
        if len(shape_x_points) == 1:
            raise ValueError("The number of points must be greater than 1.")
        return shape_x_points, shape_y_points, shape_z_points

    # def _load_into_geom(self, shape_outline_parameters: ShapeOutlineParameters, h: float):
    #     """Establish the geometry of the problem."""
    #     geom_labels = {}
    #     geom = pygmsh.built_in.Geometry()
    #     group_id = geom._TAKEN_PHYSICALGROUP_IDS
    #     curve_sets = []
    #     for shape_count, (shape_x_points, shape_y_points, shape_z_points, arc_center_points, shape_edge_types, shape_labels) in enumerate(zip(shape_outline_parameters.x_points, shape_outline_parameters.y_points, shape_outline_parameters.z_points, shape_outline_parameters.arc_center, shape_outline_parameters.edge_types, shape_outline_parameters.labels)):
    #         line_curves = []
    #         previous_point = None
    #         first_point = None
    #         shape_x_points, shape_y_points, shape_z_points = self._validate_line_points(shape_x_points, shape_y_points, shape_z_points)
    #         for x, y, z, arc_c, edge_type, label in zip(shape_x_points, shape_y_points, shape_z_points, arc_center_points, shape_edge_types, shape_labels):
    #             new_point = geom.add_point([x, y, z], h)
    #             if not first_point:
    #                 first_point = copy.copy(new_point)
    #             if previous_point:
    #                 if edge_type == 'line':
    #                     line_curves.append(geom.add_line(previous_point, new_point))
    #                 elif edge_type == 'arc':
    #                     arc_center = geom.add_point(arc_c, h)
    #                     line_curves.append(geom.add_circle_arc(previous_point, arc_center, new_point))
    #                 else:
    #                     raise ValueError("The edge type must be either 'line' or 'arc'.")
    #                 geom.add_physical_line(line_curves[-1], label=label)
    #                 geom_labels[group_id[-1]] = label
    #             previous_point = new_point
    #         if shape_edge_types[0] == 'line':
    #             line_curves.append(geom.add_line(new_point, first_point))
    #         elif shape_edge_types[0] == 'arc':
    #             arc_center = geom.add_point(arc_center_points[0], h)
    #             line_curves.append(geom.add_circle_arc(new_point, arc_center, first_point))
    #         geom.add_physical_line(line_curves[-1], label=shape_labels[0])
    #         geom_labels[group_id[-1]] = shape_labels[0]
    #         curve_sets.append(line_curves)
    #     for count, line_group in enumerate(shape_outline_parameters.line_loop_groups):
    #         line_loops = []
    #         for line_set in line_group:
    #             line_loops += curve_sets[line_set]
    #         line_group = geom.add_line_loop(line_loops)
    #         rect_surf = geom.add_plane_surface(line_group)
    #         surface_label = str(count)
    #         geom.add_physical_surface(rect_surf, label=surface_label)
    #         geom_labels[group_id[-1]] = surface_label
    #     return geom, geom_labels

    def _load_into_geom(self, shape_outline_parameters: ShapeOutlineParameters, h: float):
        geom_labels = {}

        # Use pygmsh.geo.Geometry for built-in geometry
        with pygmsh.geo.Geometry() as geom:
            gmsh.initialize()

            group_id = geom._PHYSICAL_QUEUE
            group_id = [1]
            curve_sets = []

            for shape_count, (
            shape_x_points, shape_y_points, shape_z_points, arc_center_points, shape_edge_types, shape_labels) in enumerate(
                    zip(shape_outline_parameters.x_points, shape_outline_parameters.y_points,
                        shape_outline_parameters.z_points, shape_outline_parameters.arc_center,
                        shape_outline_parameters.edge_types, shape_outline_parameters.labels)
            ):
                line_curves = []
                previous_point = None
                first_point = None

                # Validate points
                shape_x_points, shape_y_points, shape_z_points = self._validate_line_points(shape_x_points, shape_y_points,
                                                                                            shape_z_points)

                for x, y, z, arc_c, edge_type, label in zip(shape_x_points, shape_y_points, shape_z_points,
                                                            arc_center_points, shape_edge_types, shape_labels):
                    new_point = geom.add_point([x, y, z], h)

                    if first_point is None:
                        first_point = copy.copy(new_point)

                    if previous_point is not None:
                        if edge_type == 'line':
                            line_curves.append(geom.add_line(previous_point, new_point))
                        elif edge_type == 'arc':
                            arc_center = geom.add_point(arc_c, h)
                            line_curves.append(geom.add_circle_arc(previous_point, arc_center, new_point))
                        else:
                            raise ValueError("The edge type must be either 'line' or 'arc'.")

                        geom.add_physical(line_curves[-1], label=label)
                        geom_labels[group_id[-1]] = label

                    previous_point = new_point

                # Closing the loop
                if shape_edge_types[0] == 'line':
                    new_line = geom.add_line(new_point, first_point)
                    # We have todo this explicit assignment because pygmesh does an equals assert that looks at memory location and not coordinates
                    new_line.points[-1] = line_curves[0].points[0]
                    line_curves.append(new_line)
                elif shape_edge_types[0] == 'arc':
                    arc_center = geom.add_point(arc_center_points[0], h)
                    new_line = geom.add_circle_arc(new_point, arc_center, first_point)
                    # We have todo this explicit assignment because pygmesh does an equals assert that looks at memory location and not coordinates
                    new_line.points[-1] = line_curves[0].points[0]
                    line_curves.append(new_line)

                geom.add_physical(line_curves[-1], label=shape_labels[0])
                geom_labels[group_id[-1]] = shape_labels[0]
                curve_sets.append(line_curves)

            # Create surface from line loops
            for count, line_group in enumerate(shape_outline_parameters.line_loop_groups):
                line_loops = []
                for line_set in line_group:
                    line_loops += curve_sets[line_set]

                line_loop = geom.add_curve_loop(line_loops)
                rect_surf = geom.add_plane_surface(line_loop)
                surface_label = str(count)
                geom.add_physical(rect_surf, label=surface_label)
                geom_labels[group_id[-1]] = surface_label
            mesh = geom.generate_mesh()
        return mesh, geom_labels

    def plot_solution(self, p):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(p[:,0], p[:,1], edgecolor='k')
        plt.colorbar(scatter, label='Value of u')
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.title('Scatter Plot of Points Colored by u Values')
        plt.show()

    def _generate_pte_matrices(self, mesh_obj, geom_labels):
        # generate mesh
        points, cells, point_data, cell_data, field_data = mesh_obj.points, mesh_obj.cells, mesh_obj.point_data, mesh_obj.cell_data, mesh_obj.field_data
        cell_sets = mesh_obj.cell_sets
        # extract p,t,e matrices
        p = points[:, 0:2]
        t = np.zeros((cells[1].data.shape[0], 4))
        e = np.zeros((cells[0].data.shape[0], 7))
        t[:, 0:3] = cells[1].data
        e[:, 0:2] = cells[0].data

        for label, label_sets in cell_sets.items():
            flattened_elements = np.stack([el for el in label_sets if el is not None]).flatten()
            split_label = label.split(":")
            if len(split_label) == 3:
                e[flattened_elements, 4] = int(split_label[0])
                e[flattened_elements, 5] = int(split_label[1][1:])
                e[flattened_elements, 6] = int(split_label[2][1:])
            else:
                t[(np.array(flattened_elements) - np.min(flattened_elements)).astype(int), 3] = int(label)
        return np.array(p[:int(np.max(t))+1]), np.array(t.astype(int)), np.array(e.astype(int))

    def generate_mesh(self, shape_outline_parameters, h):
        """Generate the geometry and mesh."""
        mesh, geom_labels = self._load_into_geom(shape_outline_parameters, h)
        p, t, e = self._generate_pte_matrices(mesh, geom_labels)
        return p, t, e

"""Main software dev kit for managing mesh generation functions."""
from common.pydantic_models import ShapeOutlineParameters
from mesh_generation.shape_generator import ShapeGenerator
import numpy as np
import copy
import pygmsh

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

    def _load_into_geom(self, shape_outline_parameters: ShapeOutlineParameters, h: float):
        """Establish the geometry of the problem."""
        geom_labels = {}
        geom = pygmsh.built_in.Geometry()
        group_id = geom._TAKEN_PHYSICALGROUP_IDS
        curve_sets = []
        for shape_count, (shape_x_points, shape_y_points, shape_z_points, arc_center_points, shape_edge_types, shape_labels) in enumerate(zip(shape_outline_parameters.x_points, shape_outline_parameters.y_points, shape_outline_parameters.z_points, shape_outline_parameters.arc_center, shape_outline_parameters.edge_types, shape_outline_parameters.labels)):
            line_curves = []
            previous_point = None
            first_point = None
            shape_x_points, shape_y_points, shape_z_points = self._validate_line_points(shape_x_points, shape_y_points, shape_z_points)
            for x, y, z, arc_c, edge_type, label in zip(shape_x_points, shape_y_points, shape_z_points, arc_center_points, shape_edge_types, shape_labels):
                new_point = geom.add_point([x, y, z], h)
                if not first_point:
                    first_point = copy.copy(new_point)
                if previous_point:
                    if edge_type == 'line':
                        line_curves.append(geom.add_line(previous_point, new_point))
                    elif edge_type == 'arc':
                        arc_center = geom.add_point(arc_c, h)
                        line_curves.append(geom.add_circle_arc(previous_point, arc_center, new_point))
                    else:
                        raise ValueError("The edge type must be either 'line' or 'arc'.")
                    geom.add_physical_line(line_curves[-1], label=label)
                    geom_labels[group_id[-1]] = label
                previous_point = new_point
            if shape_edge_types[0] == 'line':
                line_curves.append(geom.add_line(new_point, first_point))
            elif shape_edge_types[0] == 'arc':
                arc_center = geom.add_point(arc_center_points[0], h)
                line_curves.append(geom.add_circle_arc(new_point, arc_center, first_point))
            geom.add_physical_line(line_curves[-1], label=shape_labels[0])
            geom_labels[group_id[-1]] = shape_labels[0]
            curve_sets.append(line_curves)
        for count, line_group in enumerate(shape_outline_parameters.line_loop_groups):
            line_loops = []
            for line_set in line_group:
                line_loops += curve_sets[line_set]
            line_group = geom.add_line_loop(line_loops)
            rect_surf = geom.add_plane_surface(line_group)
            surface_label = str(count)
            geom.add_physical_surface(rect_surf, label=surface_label)
            geom_labels[group_id[-1]] = surface_label
        return geom, geom_labels

    def _generate_pte_matrices(self, geom, geom_labels):
        # generate mesh
        mesh_obj = pygmsh.generate_mesh(geom)
        points, cells, point_data, cell_data, field_data = mesh_obj.points, mesh_obj.cells, mesh_obj.point_data, mesh_obj.cell_data, mesh_obj.field_data
        # extract p,t,e matrices
        p = points[:, 0:2]
        t = np.zeros((cells["triangle"].shape[0], 4))
        e = np.zeros((cells["line"].shape[0], 7))
        t[:, 0:3] = cells["triangle"]
        e[:, 0:2] = cells["line"]
        for label in field_data:
            elemnt_index = field_data[label][0]
            group_index = field_data[label][1]
            names = list(cell_data.keys())
            if names[group_index - 1] == "triangle":  # domain labels
                try:
                    t[cell_data["triangle"] \
                          ["gmsh:physical"] == elemnt_index, 3] = \
                        int(geom_labels[elemnt_index])
                except Exception as e:
                    print("think")
            elif names[group_index - 1] == "line":  # boundary labels
                label = None
                left_label = None
                right_label = None
                for item in geom_labels[elemnt_index].split(":"):
                    if item[0] == "L":
                        left_label = int(item[1:])
                    elif item[0] == "R":
                        right_label = int(item[1:])
                    else:
                        label = int(item)
                if label == None or \
                        left_label == None or \
                        right_label == None:
                    print('error, wrong geometry boundary labelling')
                index = cell_data["line"]["gmsh:physical"] == elemnt_index
                e[index, 4] = label
                e[index, 5] = left_label
                e[index, 6] = right_label
        return p, t.astype(int), e.astype(int)

    def generate_mesh(self, shape_outline_parameters, h):
        """Generate the geometry and mesh."""
        geom, geom_labels = self._load_into_geom(shape_outline_parameters, h)
        p, t, e = self._generate_pte_matrices(geom, geom_labels)
        return p, t, e

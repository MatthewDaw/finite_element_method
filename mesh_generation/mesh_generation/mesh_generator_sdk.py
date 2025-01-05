"""Main software dev kit for managing mesh generation functions."""

from mesh_generation.geom_mesh_generator import GeomMeshGenerator
from mesh_generation.adapt_mesh_generator import AdaptMesher


class MeshSDK:

    def __init__(self):
        self.geom_mesh_generator = GeomMeshGenerator()
        self.adapt_mesh_generator = AdaptMesher()

    def generate_geom_mesh(self, shape_outline_parameters, h):
        """Generate the geometry and mesh."""
        p, t, e = self.geom_mesh_generator.generate_mesh(shape_outline_parameters, h)
        return p, t, e

    def generate_adapt_mesh(self, shape_outline_parameters, h):
        """Generate the geometry and mesh."""
        p, t, e = self.adapt_mesh_generator.run_mesh(shape_outline_parameters, h)
        return p, t, e

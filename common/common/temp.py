from adaptmesh import triangulate

# from .smooth import cpt

from adaptmesh.smooth import cpt

from skfem import MeshTri

if __name__ == '__main__':


    m = triangulate([(0.0, 0.0),
                     (1.1, 0.0),
                     (1.2, 0.5),
                     (0.7, 0.6),
                     (2.0, 1.0),
                     (1.0, 2.0),
                     (0.5, 1.5),], quality=0.95, verbose=True)  # default: 0.9
import numpy as np
from common.pydantic_models import BoundaryConstraintConfig, PDECoefficients

class DiffEQParameterServer:

    def load_example_disk_setup(self):
        def c(x1, x2, sdl): return 2 + x1 + x2

        def a(x1, x2, sdl): return x1 + x2

        def f(x1, x2, sdl):
            return -8 - 6 * (x1 + x2) + (2 + x1 + x2) * \
                (np.square(x1) + np.square(x2))

        def b1(x1, x2, sdl): return x1

        def b2(x1, x2, sdl): return x2

        # boundary conditions
        def uD(x1, x2, sdl):
            f = np.zeros(x1.shape)
            I = sdl == 0
            f[I] = np.square(x1[I]) + np.square(x2[I])
            I = sdl == 1
            f[I] = np.square(x1[I]) + np.square(x2[I])
            return f

        def sigma(x1, x2, sdl):
            f = np.zeros(x1.shape)
            I = sdl == 0
            f[I] = 0
            I = sdl == 1
            f[I] = 2
            return f

        def mu(x1, x2, sdl):
            f = np.zeros(x1.shape)
            I = sdl == 0
            f[I] = 2 * (2 + x1[I] + x2[I]) * (np.square(x1[I]) + \
                                              np.square(x2[I]))
            I = sdl == 1
            f[I] = 2 * (2 + x1[I] + x2[I] + 1) * (np.square(x1[I]) + \
                                                  np.square(x2[I]))
            return f

        def exact_solution(x1, x2, sdl):
            return np.square(x1) + np.square(x2)

        bc = BoundaryConstraintConfig(
            dirichlet_boundary_labels=[1, 3],
            robin_boundary_labels=[4, 2],
            uD=uD,
            sigma=sigma,
            mu=mu
        )
        pde_coefficients = PDECoefficients(
            c=c,
            a=a,
            f=f,
            b1=b1,
            b2=b2,
            boundary_constraints=bc
        )
        return pde_coefficients, exact_solution

    def load_simple_poisson_setup(self, boundary_labels):
        """Load simple Poisson setup."""
        def c(x1, x2, sdl): return np.zeros(len(x1))-1

        def a(x1, x2, sdl): return np.zeros(len(x1))

        def f(x1, x2, sdl):
            return np.zeros(len(x1))-2

        def b1(x1, x2, sdl): return np.zeros(len(x1))

        def b2(x1, x2, sdl): return np.zeros(len(x1))

        # boundary conditions
        def uD(x1, x2, sdl):
            f = np.zeros(x1.shape)
            return f + 1

        def sigma(x1, x2, sdl):
            raise NotImplementedError("Poisson equation does not have Robin boundary conditions")

        def mu(x1, x2, sdl):
            raise NotImplementedError("Poisson equation does not have Robin boundary conditions")

        bc = BoundaryConstraintConfig(
            dirichlet_boundary_labels=boundary_labels,
            robin_boundary_labels=[],
            uD=uD,
            sigma=sigma,
            mu=mu
        )
        pde_coefficients = PDECoefficients(
            c=c,
            a=a,
            f=f,
            b1=b1,
            b2=b2,
            boundary_constraints=bc
        )
        return pde_coefficients, None

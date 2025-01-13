import numpy as np

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

from common.pydantic_models import PDECoefficients, BoundaryConstraintConfig

class FEMSolver:

    def __init__(self):
        pass

    def assemble_af(self, p, t, c, a, b1, b2, f=None):
        np_shape = p.shape[1]
        # mesh point indices
        k1 = t[0, :]
        k2 = t[1, :]
        k3 = t[2, :]
        sdl = t[3, :]  # subdomain labels
        # barycenter of the triangles
        x1 = (p[0, k1] + p[0, k2] + p[0, k3]) / 3
        x2 = (p[1, k1] + p[1, k2] + p[1, k3]) / 3
        # gradient of the basis functions, multiplied by J
        g1_x1 = p[1, k2] - p[1, k3]
        g1_x2 = p[0, k3] - p[0, k2]
        g2_x1 = p[1, k3] - p[1, k1]
        g2_x2 = p[0, k1] - p[0, k3]
        g3_x1 = p[1, k1] - p[1, k2]
        g3_x2 = p[0, k2] - p[0, k1]
        J = abs(g3_x2 * g2_x1 - g3_x1 * g2_x2)  # J=2*area
        # evaluate c , b , a on triangles barycenter
        cf = c(x1, x2, sdl)
        af = a(x1, x2, sdl)
        b1f = b1(x1, x2, sdl)
        b2f = b2(x1, x2, sdl)
        # diagonal and off diagonal elements of mass matrix
        ao = (af / 24) * J
        ad = 4 * ao  # 'exact' integration
        # ao=(af/18).*J ; ad=3*ao ; # quadrature rule
        # coefficients of the stiffness matrix
        cf = (0.5 * cf) / J
        a12 = cf * (g1_x1 * g2_x1 + g1_x2 * g2_x2) + ao
        a23 = cf * (g2_x1 * g3_x1 + g2_x2 * g3_x2) + ao
        a31 = cf * (g3_x1 * g1_x1 + g3_x2 * g1_x2) + ao
        A = []
        F = []
        if all(b1f == 0) and all(b2f == 0):  # symmetric problem
            A = sparse.coo_matrix((a12, (k1, k2)), shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((a23, (k2, k3)), shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((a31, (k3, k1)), shape=(np_shape, np_shape))
            A = A + A.transpose()
            A = A + sparse.coo_matrix((ad - a31 - a12, (k1, k1)), shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((ad - a12 - a23, (k2, k2)), shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((ad - a23 - a31, (k3, k3)), shape=(np_shape, np_shape))
        else:
            # b contributions
            b1f = b1f / 6
            b2f = b2f / 6
            bg1 = b1f * g1_x1 + b2f * g1_x2
            bg2 = b1f * g2_x1 + b2f * g2_x2
            bg3 = b1f * g3_x1 + b2f * g3_x2
            A = sparse.coo_matrix((a12 + bg2, (k1, k2)), shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((a23 + bg3, (k2, k3)), shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((a31 + bg1, (k3, k1)), shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((a12 + bg1, (k2, k1)), shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((a23 + bg2, (k3, k2)), shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((a31 + bg3, (k1, k3)), shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((ad - a31 - a12 + bg1, (k1, k1)), \
                                      shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((ad - a12 - a23 + bg2, (k2, k2)), \
                                      shape=(np_shape, np_shape))
            A = A + sparse.coo_matrix((ad - a23 - a31 + bg3, (k3, k3)), \
                                      shape=(np_shape, np_shape))
        if f != None:
            ff = f(x1, x2, sdl)
            ff = (ff / 6) * J
            k_zero = np.zeros(k1.shape)
            F = sparse.coo_matrix((ff, (k1, k_zero)), shape=(np_shape, 1))
            F = F + sparse.coo_matrix((ff, (k2, k_zero)), shape=(np_shape, 1))
            F = F + sparse.coo_matrix((ff, (k3, k_zero)), shape=(np_shape, 1))
            return A, F
        else:
            return A

    def assembling_bc(self, bc: BoundaryConstraintConfig, p, e, nargout):
        np_shape = p.shape[1]
        S = []
        M = []
        N = sparse.eye(np_shape)
        UD = []
        if (bc.robin_boundary_labels == None) and \
                (bc.sigma == None) and \
                (bc.mu == None):
            return  # all boundaries are homogeneous Robin b.c.
        e = e[:, np.logical_or(e[5, :] == 0, e[6, :] == 0)]  # boundary edges
        k = e[4, :]
        bsg = np.transpose(np.nonzero( \
            sparse.coo_matrix( \
                (np.ones(len(k)), (k, k)), shape=(np_shape, np_shape))))[:, 0]
        nbsg = len(bsg)
        # set local numeration of boundary segments
        local = np.zeros((nbsg))
        # set mixed boundary conditions
        if nargout >= 2:
            bsR = bc.robin_boundary_labels
            if len(bsR) > 0:
                if bsR == None:  # all BC are mixed
                    bsR = bsg
                local[bsR] = range(0, len(bsR))
                # find boudary edges with mixed b.c.
                eR = e[:, np.isin(k, bsR)]
                eR = eR[[0, 1, 4], :]
                k1 = eR[0, :]
                k2 = eR[1, :]  # indices of starting points and endpoints
                x1 = 0.5 * (p[0, k1] + p[0, k2])
                x2 = 0.5 * (p[1, k1] + p[1, k2])
                h = np.sqrt(np.square(p[0, k2] - p[0, k1])
                            + np.square(p[1, k2] - p[1, k1]))  # edges length
                sdl = local[eR[2, :]]
                # evaluate sigma , mu on edges barycenter
                sf = bc.sigma(x1, x2, sdl)
                mf = bc.mu(x1, x2, sdl)
                # 'exact' integration
                so = (sf / 6) * h
                sd = 2 * so
                # quadrature rule
                # so=(sf/4)*h
                # sd = so
                S = sparse.coo_matrix((so, (k1, k2)), shape=(np_shape, np_shape))
                S = S + sparse.coo_matrix((so, (k2, k1)), shape=(np_shape, np_shape))
                S = S + sparse.coo_matrix((sd, (k1, k1)), shape=(np_shape, np_shape))
                S = S + sparse.coo_matrix((sd, (k2, k2)), shape=(np_shape, np_shape))
                if nargout == 4:
                    mf = bc.mu(x1, x2, sdl)
                    mf = (mf / 2) * h
                    k_zero = np.zeros(k1.shape)
                    M = sparse.coo_matrix((mf, (k1, k_zero)), \
                                          shape=(np_shape, 1))
                    M = M + sparse.coo_matrix((mf, (k2, k_zero)), \
                                              shape=(np_shape, 1))
                    # set Dirichlet boundary conditions
        bsD = bc.dirichlet_boundary_labels
        if len(bsD) > 0:
            if bsD == None:  # all b.c. are Dirichlet
                bsD = bsg
            local[bsD] = range(0, len(bsD))
            # if all(local == 0):
            #     print('error. bsD+bsR~=number of boundary segments')
            eD = e[:, np.isin(k, bsD)]  # boudary with Dirichlet BC
            eD = eD[[0, 1, 4], :]
            sdl = local[eD[2, :]]
            # indices of start and end points
            k1 = eD[0, :]
            k2 = eD[1, :]
            iD = np.concatenate([k1, k2])  # Dirichlet points indices
            i_d = np.transpose(np.nonzero(
                sparse.coo_matrix(
                    (np.ones(len(iD)), (iD, iD)), \
                    shape=(np_shape, np_shape))))[:, 0]
            # iN - indices of non−Dirichlet points
            iN = np.ones((np_shape))
            iN[i_d] = np.zeros((len(i_d)))
            iN = np.transpose(np.nonzero(iN)).flatten()
            niN = len(iN)
            N = sparse.coo_matrix((np.ones(niN), \
                                   (iN, range(0, niN))), shape=(np_shape, niN))
            if nargout >= 3:  # evaluate UD on Dirichlet points
                UD = sparse.csr_matrix((1, np_shape))
                UD[0, k1] = bc.uD(p[0, k1], p[1, k1], sdl)
                UD[0, k2] = bc.uD(p[0, k2], p[1, k2], sdl)
                UD = UD.reshape((np_shape, 1))
        if nargout == 2:
            return N, S
        elif nargout == 3:
            return N, S, UD
        elif nargout == 4:
            return N, S, UD, M
        else:
            print('assemblingBC: Wrong number of output parameters.')

    def assembling_pde(self, p, e, t, pde_coefficients: PDECoefficients, nargout):
        # A,F,N,UD,S,M=assembling_pde(bc,p,e,t,c,a,b1,b2,f,nargout)  -
        # returns FEM matrices of the PDE problem.
        # u=assembling_pde(bc,p,e,t,c,a,b1,b2,f,nargout) -
        # returns solution to the PDE problem
        A = [];
        F = [];
        UD = [];
        S = [];
        M = []
        if nargout == 6:
            A, F = self.assemble_af(p, t, pde_coefficients.c, pde_coefficients.a, pde_coefficients.b1,
                                    pde_coefficients.b2, pde_coefficients.f)
            N, S, UD, M = self.assembling_bc(pde_coefficients.boundary_constraints, p, e, 4)
            return A, F, N, UD, S, M
        elif (nargout == 1):
            A, F = self.assemble_af(p, t, pde_coefficients.c, pde_coefficients.a, pde_coefficients.b1,
                                    pde_coefficients.b2, pde_coefficients.f)
            N, S, UD, M = self.assembling_bc(pde_coefficients.boundary_constraints, p, e, 4)
            if N.shape[1] == p.shape[1]:  # no Dirichlet BC
                A = A + S;
                F = F + M
            else:
                Nt = sparse.coo_matrix.transpose(N)
                if not isinstance(S, list):
                    A = A + S
                if not isinstance(M, list):
                    F = Nt @ ((F + M) - A @ UD)
                else:
                    F = Nt @ ((F) - A @ UD)
                A = Nt @ A @ N
            u = sparse.linalg.spsolve(A, F)
            u = (sparse.coo_matrix(u)).reshape((u.shape[0], 1))
            return (N @ u + UD)
        else:
            print('assembpde: Wrong number of output parameters.')

    def solve(self, p, t, e, pde_coefficients):
        p = np.transpose(p)
        t = np.transpose(t)
        e = np.transpose(e)
        u = self.assembling_pde(p, e, t, pde_coefficients, 1).toarray()
        return u

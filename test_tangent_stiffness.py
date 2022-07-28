from xml.dom.minicompat import NodeList
import numpy as np
import unittest
import sys

sys.path.append("..")
import pyfem


class TangentStiffness(unittest.TestCase):
    def setUp(self):
        self.creator_2d_quad = pyfem.ProblemCreator(
            nnodes_x=64, nnodes_y=64, element_type="quad"
        )
        self.creator_2d_tri = pyfem.ProblemCreator(
            nnodes_x=64, nnodes_y=64, element_type="tri"
        )
        self.creator_3d_block = pyfem.ProblemCreator(
            nnodes_x=8, nnodes_y=8, nnodes_z=8, element_type="block"
        )
        return

    def run_tangent_stiffness(self, creator, quadrature, basis):
        """
        Test the sensitivity of buckling modes w.r.t. nodal variable X
        """
        (conn, X, dof_fixed, nodal_force,) = creator.create_linear_elasticity_problem()
        model = pyfem.LinearElasticity(
            X, conn, dof_fixed, None, nodal_force, quadrature, basis, p=5.0
        )

        nnodes = X.shape[0]
        ndof = X.shape[0] * X.shape[1]

        np.random.seed(0)
        rho = np.random.rand(nnodes)

        G = model.compute_tangent_stiffness(rho)
        K = model.compute_jacobian(rho)

        # Check that G is the right shape
        self.assertTupleEqual(G.shape, (ndof, ndof))

        # Check that G is symmetric
        # There's surely a better way to do this but this is what I came up with
        phi = np.random.rand(ndof)
        psi = np.random.rand(ndof)

        phiGpsi = phi @ G @ psi
        psiGphi = psi @ G @ phi
        self.assertAlmostEqual((phiGpsi - psiGphi) / phiGpsi, 0, delta=1e-12)

        # h = 1e-30

        # n_eig = 12

        ## Compute eigenvalue sensitivities
        # dmudrho = model._compute_buck_eig_sens(rho, n_eig)

        # Compute eigenvalue sensitivities via complex step

        # for i in range(model.nelems):
        #    p = np.zeros(ndof)
        #    p[i] = 1
        #    mu = model.compute_buck_eig(rho + 1j * p * h)
        #    dmudrhoi_cs = mu.imag / h

        #    self.assertAlmostEqual(np.linalg.norm(dmudrho[i] - dmudrhoi_cs) / np.linalg.norm(dmudrho[i]), 0.0, delta=1e-12)

    # def test_2d_quad(self):
    #    quadrature = pyfem.QuadratureBilinear2D()
    #    basis = pyfem.BasisBilinear2D(quadrature)
    #    self.run_buck_eig_sens(self.creator_2d_quad, quadrature, basis)
    #    return

    # def test_2d_tri(self):
    #    quadrature = pyfem.QuadratureTriangle2D()
    #    basis = pyfem.BasisTriangle2D(quadrature)
    #    self.run_buck_eig_sens(self.creator_2d_tri, quadrature, basis)
    #    return

    # def test_3d_block(self):
    #    quadrature = pyfem.QuadratureBlock3D()
    #    basis = pyfem.BasisBlock3D(quadrature)
    #    self.run_buck_eig_sens(self.creator_3d_block, quadrature, basis)
    #    return

    def test_tangent_stiffness(self):
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)
        self.run_tangent_stiffness(self.creator_2d_quad, quadrature, basis)


if __name__ == "__main__":
    unittest.main()

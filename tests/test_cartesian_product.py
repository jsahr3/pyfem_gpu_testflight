from xml.dom.minicompat import NodeList
import numpy as np
import unittest
import sys

sys.path.append("..")
import pyfem


class Block3DBasisEquivalency(unittest.TestCase):
    def setUp(self):

        # Declare and save the bases required to construct the composite bases
        self.Linear1D = pyfem.BasisLinear1D(pyfem.QuadratureLinear1D())
        self.Bilinear2D = pyfem.BasisBilinear2D(pyfem.QuadratureBilinear2D())
        self.Block3D = pyfem.BasisBlock3D(pyfem.QuadratureBlock3D())
        return

    def BasisEquivalency(self, Basis):

        # Sanity checks
        self.assertEqual(self.Block3D.ndims, Basis.ndims)
        self.assertEqual(self.Block3D.nnodes_per_elem, Basis.nnodes_per_elem)
        self.assertEqual(self.Block3D.nquads, Basis.nquads)

        ndims = self.Block3D.ndims
        nnodes_per_elem = self.Block3D.nnodes_per_elem
        nquads = self.Block3D.nquads
        shape_fun_1 = self.Block3D.eval_shape_fun()
        shape_fun_2 = Basis.eval_shape_fun()
        shape_fun_deriv_1 = self.Block3D.eval_shape_fun_deriv()
        shape_fun_deriv_2 = Basis.eval_shape_fun_deriv()
        pts_1 = self.Block3D.quadrature.get_pt()
        pts_2 = Basis.quadrature.get_pt()
        weights_1 = self.Block3D.quadrature.get_weight()
        weights_2 = Basis.quadrature.get_weight()

        # For each quadrature point in the known good quadrature:
        for q_1 in range(nquads):

            # Find the closest quadrature point in the test quadrature
            q_2_best = 0
            for q_2 in range(nquads):
                # fmt: off
                if (
                    np.linalg.norm(pts_2[q_2] - pts_1[q_1]) <
                    np.linalg.norm(pts_2[q_2_best] - pts_1[q_1])
                ):
                # fmt: on
                    q_2_best = q_2
            q_2 = q_2_best

            # Check that that quadrature point really is close
            for d in range(ndims):
                self.assertAlmostEqual(pts_1[q_1, d], pts_2[q_2, d], delta=1e-12)

            # Check that that quadrature point has the same weight
            self.assertAlmostEqual(weights_1[q_1], weights_2[q_2], delta=1e-12)

            # For each node in the known good basis:
            for n_1 in range(nnodes_per_elem):

                # Find the node with the closest shape derivs in the test basis
                n_2_best = 0
                for n_2 in range(nnodes_per_elem):
                    if (
                        np.linalg.norm(shape_fun_deriv_2[q_2, n_2, :] - shape_fun_deriv_1[q_1, n_1, :]) <
                        np.linalg.norm(shape_fun_deriv_2[q_2, n_2_best, :] - shape_fun_deriv_1[q_1, n_1, :])
                    ):
                        n_2_best = n_2
                n_2 = n_2_best

                # Check that that node really has close shape derivs
                for d in range(ndims):
                    self.assertAlmostEqual(shape_fun_deriv_1[q_1, n_1, d], shape_fun_deriv_2[q_2, n_2, d], delta=1e-12)

                # Check that the shape functions are equal
                self.assertAlmostEqual(shape_fun_1[q_1, n_1], shape_fun_2[q_2, n_2], delta=1e-12)

        return

    def test_lin_lin_lin(self):
        self.BasisEquivalency(
            pyfem.BasisProduct(self.Linear1D, self.Linear1D, self.Linear1D)
        )
        return

    def test_lin_bilin(self):
        self.BasisEquivalency(pyfem.BasisProduct(self.Linear1D, self.Bilinear2D))
        return

    def test_bilin_lin(self):
        self.BasisEquivalency(pyfem.BasisProduct(self.Bilinear2D, self.Linear1D))
        return


if __name__ == "__main__":
    unittest.main()

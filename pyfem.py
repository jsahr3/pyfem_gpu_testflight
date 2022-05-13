import numpy as np
from scipy import sparse
from time import perf_counter_ns
from abc import ABC, abstractmethod
import argparse


def time_this(func):
    """
    Decorator: time the execution of a function
    """
    tab = "    "

    def wrapper(*args, **kwargs):
        print(f"[timer] {tab*time_this.counter}{func.__name__:s}() called")
        time_this.counter += 1
        t0 = perf_counter_ns()
        ret = func(*args, **kwargs)
        t1 = perf_counter_ns()
        time_this.counter -= 1
        print(
            f"[timer] {tab*time_this.counter}{func.__name__:s}() return",
            f"({(t1 - t0) / 1e6:.2f} ms)",
        )
        return ret

    return wrapper


time_this.counter = 0


class QuadratureBase(ABC):
    """
    Abstract class for quadrature object
    """

    def __init__(self, pts, weights):
        self.pts = pts
        self.weights = weights
        self.nquads = pts.shape[0]
        return

    def get_nquads(self):
        """
        Get number of quadrature points per element
        """
        return self.nquads

    @abstractmethod
    def get_pt(self, idx=None):
        """
        Query the <idx>-th quadrature point (xi, eta) or (xi, eta, zeta) based
        on quadrature type, if idx is None, return all quadrature points as a
        list
        """
        if idx:
            return self.pts[idx]
        else:
            return self.pts

    @abstractmethod
    def get_weight(self, idx=None):
        """
        Query the weight of <idx>-th quadrature point, if idx is None, return
        all quadrature points as a list
        """
        if idx:
            return self.weights[idx]
        else:
            return self.weights


class BasisBase(ABC):
    """
    Abstract class for the element basis function
    """

    def __init__(self, quadrature):
        """
        Inputs:
            quadrature: object of type QuadratureBase
        """
        self.quadrature = quadrature
        return

    @abstractmethod
    def eval_shape_fun(self, N):
        """
        Evaluate the shape function values at quadrature points

        Outputs:
            N: shape function values, (nquads, nnodes_per_elem)
        """
        return

    @abstractmethod
    def eval_shape_fun_deriv(self, Nderiv):
        """
        Evaluate the shape function derivatives at quadrature points

        Outputs:
            Nderiv: shape function derivatives w.r.t. local coordinate,
                    (nquads, nnodes_per_elem, ndims)
        """
        return


class PhysicalModelBase(ABC):
    """
    Abstract class for the physical problem to be solved by the finite element
    method
    TODO: how to generalize this class?
    """

    def __init__(self, nvars_per_node):
        self.nvars_per_node = nvars_per_node
        return

    def get_nvars_per_node(self):
        return self.nvars_per_node

    @abstractmethod
    def eval_element_rhs(self):
        return

    @abstractmethod
    def eval_element_jacobian(self):
        return


class QuadratureBilinear2D(QuadratureBase):
    def __init__(self):
        pts = np.array(
            [
                # fmt: off
                [-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                [ 1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                [ 1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
                [-1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
            ]
        )
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        super().__init__(pts, weights)
        return

    def get_pt(self, idx=None):
        return super().get_pt(idx)

    def get_weight(self, idx=None):
        return super().get_weight(idx)


class BasisBilinear2D(BasisBase):
    def __init__(self, quadrature):
        super().__init__(quadrature)
        return

    def eval_shape_fun(self, N):
        quad_pts = self.quadrature.get_pt()
        N[...] = np.array(
            list(
                map(
                    lambda qpt: [
                        0.25 * (1.0 - qpt[0]) * (1.0 - qpt[1]),
                        0.25 * (1.0 + qpt[0]) * (1.0 - qpt[1]),
                        0.25 * (1.0 + qpt[0]) * (1.0 + qpt[1]),
                        0.25 * (1.0 - qpt[0]) * (1.0 + qpt[1]),
                    ],
                    quad_pts,
                )
            )
        )
        return

    def eval_shape_fun_deriv(self, Nderiv):
        quad_pts = self.quadrature.get_pt()
        Nderiv[..., 0] = np.array(
            list(
                map(
                    lambda qpt: [
                        # fmt: off
                        -0.25 * (1.0 - qpt[1]),
                         0.25 * (1.0 - qpt[1]),
                         0.25 * (1.0 + qpt[1]),
                        -0.25 * (1.0 + qpt[1]),
                    ],
                    quad_pts,
                )
            )
        )
        Nderiv[..., 1] = np.array(
            list(
                map(
                    lambda qpt: [
                        # fmt: off
                        -0.25 * (1.0 - qpt[0]),
                        -0.25 * (1.0 + qpt[0]),
                         0.25 * (1.0 + qpt[0]),
                         0.25 * (1.0 - qpt[0]),
                    ],
                    quad_pts,
                )
            )
        )
        return


class LinearPoisson2D(PhysicalModelBase):
    """
    The 2-dimensional Poisson equation

    Equation:
    -∆u = g in Ω

    Boundary condition:
    u = u0 on ∂Ω

    Weak form:
    ∫ ∇u ∇v dΩ = ∫gv dΩ for test function
    """

    def __init__(self):
        self.fun_vals = None
        nvars_per_node = 1
        super().__init__(nvars_per_node)
        return

    def _eval_gfun(self, Xq, fun_vals):
        xvals = Xq[..., 0]
        yvals = Xq[..., 1]
        fun_vals[...] = xvals * (xvals - 5.0) * (xvals - 10.0) * yvals * (yvals - 4.0)
        return

    def eval_element_rhs(self, quadrature, N, detJq, Xq, rhs_e):
        """
        Evaluate element-wise rhs vectors:

            rhs_e = ∑ detJq wq (gN)_q
                    q

        Inputs:
            quadrature (QuadratureBase)
            N: shape function values, (nnodes_per_elem, nquads)
            detJq: Jacobian determinants, (nelems, nquads)
            Xq: quadrature location in global coordinates, (nelems,
                nquads, ndims)

        Outputs:
            rhs_e: element-wise rhs, (nelems, nnodes_per_elem * nvars_per_node)
        """
        # Allocate memory for quadrature function values
        if self.fun_vals is None:
            self.fun_vals = np.zeros(Xq.shape[0:-1])

        # Evaluate function g
        self._eval_gfun(Xq, self.fun_vals)

        # Compute element rhs
        wq = quadrature.get_weight()
        rhs_e[...] = np.einsum("ik,k,jk,ik -> ij", detJq, wq, N, self.fun_vals)
        return

    def eval_element_jacobian(self, quadrature, Ngrad, detJq, Ke):
        """
        Evaluate element-wise Jacobian matrices

            Ke = ∑ detJq wq ( NxNxT + NyNyT)_q
                 q

        Inputs:
            quadrature (QuadratureBase)
            Ngrad: shape function gradient, (nelems, nquads, nnodes_per_elem,
                   ndims)
            detJq: Jacobian determinants, (nelems, nquads)

        Outputs:
            Ke: element-wise Jacobian matrix, (nelems, nnodes_per_elem *
                nvars_per_node, nnodes_per_elem * nvars_per_node)
        """
        wq = quadrature.get_weight()
        Ke[...] = np.einsum("iq,q,iqjl,iqkl -> ijk", detJq, wq, Ngrad, Ngrad)
        return


class Assembler:
    def __init__(
        self,
        conn,
        X,
        quadrature: QuadratureBase,
        basis: BasisBase,
        model: PhysicalModelBase,
    ):
        # Prepare and take inputs
        self.conn = np.array(conn)  # connectivity: nelems * nnodes_per_elem
        self.X = np.array(X)  # nodal locations: nnodes * ndims
        self.quadrature = quadrature
        self.basis = basis
        self.model = model  # physical model

        # Get dimension information
        self.nelems = conn.shape[0]
        self.nnodes_per_elem = conn.shape[1]
        self.ndims = X.shape[1]
        self.nquads = self.quadrature.get_nquads()
        self.nvars_per_node = self.model.get_nvars_per_node()

        # Compute non-zero indices
        self.i, self.j = self._compute_nz_pattern()

        # Allocate memory for the element-wise data
        self.Xe = np.zeros(
            (self.nelems, self.nnodes_per_elem, self.ndims)
        )  # nodal locations
        self.Ue = np.zeros(
            (self.nelems, self.nnodes_per_elem, self.nvars_per_node)
        )  # solution variable
        self.rhs_e = np.zeros(
            (self.nelems, self.nnodes_per_elem * self.nvars_per_node)
        )  # rhs vectors for each element
        self.Ke = np.zeros(
            (
                self.nelems,
                self.nnodes_per_elem * self.nvars_per_node,
                self.nnodes_per_elem * self.nvars_per_node,
            )
        )

        # Scatter X -> Xe
        self._scatter_node_to_elem(self.X, self.Xe)

        # Allocate memory for quadrature data
        self.Xq = np.zeros((self.nelems, self.nquads, self.ndims))  # nodal locations
        self.Jq = np.zeros(
            (self.nelems, self.nquads, self.ndims, self.ndims)
        )  # Jacobian transformations
        self.invJq = np.zeros(
            (self.nelems, self.nquads, self.ndims, self.ndims)
        )  # inverse of Jacobian transformations
        self.detJq = np.zeros(
            (self.nelems, self.nquads)
        )  # determinants of Jacobian transformations
        self.N = np.zeros((self.nquads, self.nnodes_per_elem))  # shape function values
        self.Nderiv = np.zeros(
            (self.nquads, self.nnodes_per_elem, self.ndims)
        )  # shape function derivatives w.r.t. local coordinates
        self.Ngrad = np.zeros(
            (self.nelems, self.nquads, self.nnodes_per_elem, self.ndims)
        )  # shape gradient w.r.t. global coordinates

        # Global Jacobian (stiffness) matrix and rhs to be created
        self.rhs = None
        self.K = None
        return

    @time_this
    def _compute_nz_pattern(self):
        """
        Compute the non-zero coordinates
        """
        # Compute non-zero pattern (i, j)
        i = []
        j = []
        val = []
        for index in range(self.nelems):
            for ii in self.conn[index, :]:
                for jj in self.conn[index, :]:
                    i.append(ii)
                    j.append(jj)
                    val.append(0)
        return i, j

    @time_this
    def _scatter_node_to_elem(self, data, data_e):
        """
        Scatter nodal quantities to elements: data -> data_e

        Inputs:
            data: nodal data, (nnodes, X)

        Outputs:
            data_e: element-wise data, (nelems, nnodes_per_elem, X)
        """
        data_e[...] = data[self.conn]
        return

    @time_this
    def _compute_jtrans(self, Xe, Nderiv, Jq):
        """
        Compute the Jacobian transformation, inverse and determinant

        Inputs:
            Xe: element nodal location
            Nderiv: derivative of basis w.r.t. local coordinate

        Outputs:
            Jq: the Jacobian matrix on quadrature
        """
        Jq[...] = np.einsum("ijk, mjn -> imkn", Xe, Nderiv)
        return

    @time_this
    def _compute_jdet(self, Jq, detJq):
        """
        Compute the determinant given Jacobian.

        Inputs:
            Jq: the Jacobian matrices for each element and each quadrature point

        Outputs:
            detJq: the determinant of the Jacobian matrices
        """
        if self.ndims == 2:
            detJq[...] = Jq[..., 0, 0] * Jq[..., 1, 1] - Jq[..., 0, 1] * Jq[..., 1, 0]
        else:
            raise NotImplementedError()
        return

    @time_this
    def _compute_elem_interp(self, N, data_e, data_q):
        """
        Interpolate the quantities at quadrature points:
        data_e -> data_q

        Inputs:
            N: basis values, (nquads, nnodes_per_elem)
            data_e: element data, (nelems, nnodes_per_elem, X)

        Outputs:
            data_q: quadrature data, (nelems, nquads, X)
        """
        data_q[...] = np.einsum("jl, ilk -> ijk", N, data_e)
        return

    @time_this
    def _compute_basis_grad(self, Jq, detJq, Nderiv, invJq, Ngrad):
        """
        Compute the derivatives of basis function with respect to global
        coordinates on quadratures

        Inputs:
            Jq: Jacobian transformation on quadrature, (nelems, nquads, ndims,
                ndims)
            detJq: the determinant of the Jacobian matrices, (nelems, nquads)
            Nderiv: shape function derivatives, (nquads, nnodes_per_elem, ndims)

        Outputs:
            invJq: Jacobian inverse, by-product, (nelems, nquads, ndims, ndims)
            Ngrad: shape function gradient, (nelems, nquads, nnodes_per_elem,
                   ndims)
        """
        # Compute Jacobian inverse
        if self.ndims == 2:
            invJq[..., 0, 0] = Jq[..., 1, 1] / detJq
            invJq[..., 0, 1] = -Jq[..., 0, 1] / detJq
            invJq[..., 1, 0] = -Jq[..., 1, 0] / detJq
            invJq[..., 1, 1] = Jq[..., 0, 0] / detJq
        else:
            raise NotImplementedError
        Ngrad[...] = np.einsum("jkm, ijml -> ijkl", Nderiv, invJq)
        return

    @time_this
    def _assemble_rhs(self, quadrature, rhs_e, rhs):
        """
        Assemble the global rhs given element-wise rhs: rhs_e -> rhs

        Inputs:
            quadrature: the quadrature object
            rhs_e: element-wise rhs, (nelems, nnodes_per_elem * nvars_per_node)

        Outputs:
            rhs: global rhs, (nnodes * nvars_per_node, )
        """
        for n in range(quadrature.get_nquads()):
            np.add.at(rhs[:], self.conn[:, n], rhs_e[:, n])
        return

    @time_this
    def _assemble_jacobian(self, Ke):
        """
        Assemble global K matrix
        """
        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        return K.tocsr()

    @time_this
    def compute_rhs(self, rhs):
        """
        Compute the element rhs vectors and assemble the global rhs

        Outputs:
            rhs: global right-hand-side vector
        """
        # Compute shape function and derivatives
        self.basis.eval_shape_fun(self.N)
        self.basis.eval_shape_fun_deriv(self.Nderiv)

        # Compute Jacobian transformation -> Jq
        self._compute_jtrans(self.Xe, self.Nderiv, self.Jq)

        # Compute Jacobian determinant
        self._compute_jdet(self.Jq, self.detJq)

        # Compute element mapping -> Xq
        self._compute_elem_interp(self.N, self.Xe, self.Xq)

        # Compute element rhs vector -> rhs_e
        self.model.eval_element_rhs(
            self.quadrature, self.N, self.detJq, self.Xq, self.rhs_e
        )

        # Assemble the global rhs vector -> rhs
        self._assemble_rhs(self.quadrature, self.rhs_e, rhs)
        return

    @time_this
    def compute_jacobian(self):
        """
        Compute the element Jacobian (stiffness) matrices Ke and assemble the
        global K

        Return:
            K: (sparse) global Jacobian matrix
        """
        # Compute Jacobian derivatives
        self.basis.eval_shape_fun_deriv(self.Nderiv)

        # Compute Jacobian transformation -> Jq
        self._compute_jtrans(self.Xe, self.Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        self._compute_jdet(self.Jq, self.detJq)

        # Compute shape function gradient -> (invJq), Ngrad
        self._compute_basis_grad(
            self.Jq, self.detJq, self.Nderiv, self.invJq, self.Ngrad
        )

        # Compute element Jacobian -> Ke
        self.model.eval_element_jacobian(
            self.quadrature, self.Ngrad, self.detJq, self.Ke
        )

        return self._assemble_jacobian(self.Ke)


def create_problem(n=3):
    m = n * 4
    nelems = m * n
    nnodes = (m + 1) * (n + 1)
    y = np.linspace(0, 4, n + 1)
    x = np.linspace(0, 10, m + 1)
    nodes = np.arange(0, (n + 1) * (m + 1)).reshape((n + 1, m + 1))

    # Set the node locations
    X = np.zeros((nnodes, 2))
    for j in range(n + 1):
        for i in range(m + 1):
            X[i + j * (m + 1), 0] = x[i]
            X[i + j * (m + 1), 1] = y[j]

    # Set the connectivity
    conn = np.zeros((nelems, 4), dtype=int)
    for j in range(n):
        for i in range(m):
            conn[i + j * m, 0] = nodes[j, i]
            conn[i + j * m, 1] = nodes[j, i + 1]
            conn[i + j * m, 2] = nodes[j + 1, i + 1]
            conn[i + j * m, 3] = nodes[j + 1, i]

    # Set the constrained degrees of freedom at each node
    bcs = []
    for j in range(n):
        bcs.append(nodes[j, 0])
        bcs.append(nodes[j, -1])

    return nelems, nnodes, conn, X, bcs


@time_this
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()

    # Create mesh
    nelems, nnodes, conn, X, bcs = create_problem(n=args.n)

    # Old code
    from ref_linear_poisson import Poisson, gfunc

    poisson = Poisson(conn, X, bcs, gfunc)

    # Refactored code
    quadrature = QuadratureBilinear2D()
    basis = BasisBilinear2D(quadrature)
    model = LinearPoisson2D()
    assembler = Assembler(conn, X, quadrature, basis, model)

    # Check rhs
    rhs_old = poisson._compute_rhs(gfunc)
    rhs = np.zeros(nnodes)
    assembler.compute_rhs(rhs)
    print("Check global rhs:")
    print(np.max(rhs_old))
    print(np.max(rhs_old - rhs))

    # Check K
    K_old = poisson.assemble_jacobian()
    K = assembler.compute_jacobian()
    print("Check global K:")
    print(np.max(K_old))
    print(np.max(K_old - K))


if __name__ == "__main__":
    main()
"""Solve a model poisson problem with Dirichlet boundary conditions
using the finite element method.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from __future__ import division
from fe_utils import *
import numpy as np
from numpy import sin, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser


def assemble(fs, f):
    """Assemble the finite element system for the Poisson problem given
    the function space in which to solve and the right hand side
    function."""

    # The Poisson problem can be solved similarly to the Helmholtz problem
    # except we dont have the integral(uv) term in the weak form, corresponding
    # to ignoring the integral(phi_i * phi_j) in the expression for A_{ij}.
    # Therefore we procede as before:

    # Create an appropriate (complete) quadrature rule.
    # We use 2 * (degree - 1) because we need to integrate
    # grad phi_i * grad phi_j.
    quadrature_rule = gauss_quadrature(
        fs.element.cell, 2*(fs.element.degree - 1))

    # Tabulate the basis functions and their gradients
    # at the quadrature points:
    basis_values = fs.element.tabulate(quadrature_rule.points)
    basis_grads = fs.element.tabulate(quadrature_rule.points, grad=True)
    qweights = quadrature_rule.weights

    # Create the left hand side matrix and right hand side vector.
    A = sp.lil_matrix((fs.node_count, fs.node_count))
    rhs = np.zeros(fs.node_count)

    # Compute the Jacobians for each cell:
    jacobians = fs.mesh.jacobian(...)
    # And their determinants:
    detJs = np.abs(np.linalg.det(jacobians))
    # And inverses:
    invJs = np.linalg.inv(jacobians)
    # Compute the value of the function at each cell node:
    cell_values = f.values[fs.cell_nodes[:, :].ravel()] \
        .reshape(fs.cell_nodes.shape)

    # Find the integral(grad phi_i * grad phi_j) term of A_{ij}
    # (from the expanded form in 6.79): 
    A_terms = np.einsum('c,cyx,qiy,czx,qjz,q->cij', 
        detJs, invJs, basis_grads, invJs, basis_grads, qweights, 
        optimize=True)
    # Find the non-zero values of the right hand side (from 6.72):
    l_terms = np.einsum('c,qi,ck,qk,q->ci', 
        detJs, basis_values, cell_values, basis_values, qweights, 
        optimize=True)

    # For each cell:
    for c in range(fs.mesh.entity_counts[-1]):
        # Insert the right hand side terms at the right location:
        rhs[fs.cell_nodes[c, :]] += l_terms[c, :]
        # Insert the left hand side terms at the right location (using
        # np.ix_ to index a submatrix not a diagonal):
        A[np.ix_(fs.cell_nodes[c, :], fs.cell_nodes[c, :])] += A_terms[c, :, :]

    # Process the boundary conditions according to 7.1:
    for node in boundary_nodes(fs):
        rhs[node] = 0
        A[node, :] = 0
        A[node, node] = 1

    return A, rhs


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1.
        else:
            return 0.

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)


def solve_poisson(degree, resolution, analytic=False, return_error=False):
    """Solve a model Poisson problem on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: sin(4*pi*x[0])*x[1]**2*(1.-x[1])**2)

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: (16*pi**2*(x[1] - 1)**2*x[1]**2 - 2*(x[1] - 1)**2 -
                             8*(x[1] - 1)*x[1] - 2*x[1]**2) * sin(4*pi*x[0]))

    # Assemble the finite element system.
    A, l = assemble(fs, f)

    # Create the function to hold the solution.
    u = Function(fs)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csr_matrix(A)
    u.values[:] = splinalg.spsolve(A, l)

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return u, error

if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve a Poisson problem on the unit square.""")
    parser.add_argument("--analytic", action="store_true",
        help="Plot the analytic solution instead of solving" \
        " the finite element problem.")
    parser.add_argument("--error", action="store_true",
        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
        help="The number of cells in each direction on the mesh.")
    parser.add_argument("degree", type=int, nargs=1,
        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    degree = args.degree[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_poisson(degree, resolution, analytic, plot_error)

    u.plot()

# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.
    """

    if cell is ReferenceInterval:
        # Create (degree + 1) choose 1 == degree + 1 equally spaced 
        # points over [0, 1], then add an extra axis so the shape is
        # (degree + 1, 1) not (degree + 1,)
        return np.linspace(0, 1, degree + 1)[:, None]
    elif cell is ReferenceTriangle:
        # Create the equivalent 1D points:
        one_d = np.linspace(0, 1, degree + 1)
        # Find the Cartesian product of one_d with itself to generate
        # equally spaced points in a square:
        pairs = np.dstack(np.meshgrid(one_d, one_d)).reshape(-1, 2)
        # Return just the (degree + 2)*(degree + 1)/2 == (degree + 2) choose 2
        # points that are in the reference triangle:
        return pairs[pairs[:, 0] + pairs[:, 1] <= 1]


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`
    """

    if cell.dim == 1:
        if not grad:
            # For 1D, numpy has a built-in function for it (remembering
            # to flatten the points to a 1D array, and adjust the ordering):
            return np.vander(points[:, 0], degree + 1, increasing=True)
        else:
            # For the gradient case, we must build the matrix ourselves:
            V = np.zeros((points.shape[0], degree + 1, 1))
            # For each monomial, calculate the gradient (skipping x^0):
            for deg in range(1, degree + 1):
                # d/dx x^n = n*x^(n - 1):
                V[:, deg, 0] = deg * points[:, 0] ** (deg - 1)

            return V
    elif cell.dim == 2:
        # For 2D we must do it manually.
        # Create an output array of the right shape:
        if not grad:
            V = np.zeros((points.shape[0], (degree + 2)*(degree + 1)//2))
        else:
            V = np.zeros((points.shape[0], (degree + 2)*(degree + 1)//2, 2))
            
        # For each monomial (total) degree we need:
        idx = 0
        for deg in range(degree + 1):
            # For each combination of degrees that sum to that:
            for ydeg in range(deg + 1):
                if not grad:
                    # The next column is y^k * x^(d - k):
                    V[:, idx] = points[:, 1]**ydeg * points[:, 0]**(deg - ydeg)
                else:
                    # The next column is k*y^(k - 1) * x^(d - k) and 
                    # y^k * (d - k) * x ^ (d - k - 1).
                    # Skip the y^0 for the y derivatives:
                    if ydeg > 0:
                        V[:, idx, 1] = ydeg * points[:, 1]**(ydeg - 1) \
                            * points[:, 0]**(deg - ydeg)
                    # Skip the x^0 for the x derivatives:
                    if ydeg < deg:
                        V[:, idx, 0] = points[:, 1]**ydeg \
                            * (deg - ydeg) * points[:, 0]**(deg - ydeg - 1)
                # Move over to the next column:
                idx += 1

        return V


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with 
            entity `(d, i)`.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Find the basis for values polynomials of the given
        # degree evaluated on the nodes:
        poly_basis = vandermonde_matrix(cell, degree, nodes)
        # Invert that to get the basis for the coefficients of
        # the polynomials given the node values:
        self.basis_coefs = np.linalg.inv(poly_basis)

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.
        """

        if not grad:
            # A basis for the values of polynomials evaluated at the
            # given list of points:
            poly_points = vandermonde_matrix(
                self.cell, self.degree, points, grad=False)
            # Compute the values of the basis functions at those points:
            return poly_points @ self.basis_coefs
        else:
            # A basis for the gradients of polynomials evaluated at the
            # given list of points:
            poly_points = vandermonde_matrix(
                self.cell, self.degree, points, grad=True)
            # Compute the gradients of the basis functions at those points:
            # (essentially a matrix product but the elements of the
            # matrix are vectors)
            return np.einsum("ijk,jl->ilk", poly_points, self.basis_coefs)


    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.
        """

        return np.apply_along_axis(fn, 1, self.nodes)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        """

        # The nodes are given by the Lagrange points for that
        # degree and cell combination:
        nodes = lagrange_points(cell, degree)

        # Setup an empty entity nodes dict:
        entity_nodes = { 
            d: { i: [] for i in cell.topology[d] } for d in cell.topology 
        }

        # Since the lagrange_points function produces points 
        # in the right order, we can go ahead and just add them as they 
        # appear to whatever entity they belong too:
        entities = [(d, i) for d in cell.topology for i in cell.topology[d]]
        for idx in range(nodes.shape[0]):
            # Look through each entity it coule be in:
            for (d, i) in entities:
                # If the point is in this entity:
                if cell.point_in_entity(nodes[idx, :], (d, i)):
                    # Append it to the list, 
                    entity_nodes[d][i].append(idx)
                    # Then break out of the loop so we don't add it
                    # to more than one list.
                    break

        # Setup the basis coefficients and everything else:
        super(LagrangeElement, self).__init__(
            cell, degree, nodes, entity_nodes)

from scipy.spatial import Delaunay
import numpy as np
import itertools
from .finite_elements import LagrangeElement
from .reference_elements import ReferenceTriangle, ReferenceInterval


class Mesh(object):
    """A one or two dimensional mesh composed of intervals or triangles
    respectively."""
    def __init__(self, vertex_coords, cell_vertices):
        """
        :param vertex_coords: an vertex_count x dim array of the coordinates of
          the vertices in the mesh.
        :param cell_vertices: an cell_count x (dim+1) array of the
          indices of the vertices of which each cell is made up.
        """

        self.dim = vertex_coords.shape[1]
        """The geometric and topological dimension of the mesh. Immersed
        manifolds are not supported.
        """

        if self.dim not in (1, 2):
            raise ValueError("Only 1D and 2D meshes are supported")

        self.vertex_coords = vertex_coords
        """The coordinates of all the vertices in the mesh."""

        self.cell_vertices = np.sort(cell_vertices)
        """The indices of the vertices incident to cell."""

        if self.dim == 2:
            self.edge_vertices = np.array(list(set(tuple(sorted(e))
                for t in cell_vertices
                for e in itertools.combinations(t, 2))))
            """The indices of the vertices incident to edge (only for 2D
            meshes)."""

            # Invert self.edge_vertices so that it is possible to look up
            # the edge index given the vertex indices.
            edge_dict = {tuple(e): i
                         for i, e_ in enumerate(self.edge_vertices)
                         for e in (e_, reversed(e_))}

            # List the local vertex indices associated with
            # each local edge index.
            local_edge_vertices = np.array([[1, 2], [0, 2], [0, 1]])

            self.cell_edges = np.fromiter(
                (edge_dict[tuple(t.take(local_edge_vertices[e]))]
                 for t in self.cell_vertices
                 for e in range(3)),
                dtype=np.int32,
                count=self.cell_vertices.size).reshape((-1, 3))
            """The indices of the edges incident to each cell (only for 2D
            meshes)."""

        if self.dim == 2:
            self.entity_counts = np.array((vertex_coords.shape[0],
                                           self.edge_vertices.shape[0],
                                           self.cell_vertices.shape[0]))
            """The number of entities of each dimension in the mesh. So
            :attr:`entity_counts[0]` is the number of vertices in the
            mesh."""
        else:
            self.entity_counts = np.array((vertex_coords.shape[0],
                                           self.cell_vertices.shape[0]))

        #: The :class:`~.reference_elements.ReferenceCell` of which this
        #: :class:`Mesh` is composed.
        self.cell = (0, ReferenceInterval, ReferenceTriangle)[self.dim]



        # Construct a linear Lagrange basis on the reference cell:
        linear_basis = LagrangeElement(self.cell, 1)
        # Create the (0, 0) vertex we want to evaluate the gradient at:
        zero_vertex = np.zeros((1, self.dim))
        # Precompute the gradient of the linear basis at zero to speed up
        # Jacobian calculations later:
        
        #: The gradient of the linear Lagrange basis for the 
        #: :class:`~.reference_elements.ReferenceCell` of this :class:`Mesh`
        #: evaluated at (0, 0).
        self.grad_linear_basis = linear_basis.tabulate(
            zero_vertex, grad=True)[0, ...]

    def adjacency(self, dim1, dim2):
        """Return the set of `dim2` entities adjacent to each `dim1`
        entity. For example if `dim1==2` and `dim2==1` then return the list of
        edges (1D entities) adjacent to each triangle (2D entity).

        The return value is a rank 2 :class:`numpy.array` such that
        ``adjacency(dim1, dim2)[e1, :]`` is the list of dim2 entities
        adjacent to entity ``(dim1, e1)``.

        This operation is only defined where `self.dim >= dim1 > dim2`.

        This method is simply a more systematic way of accessing
        :attr:`edge_vertices`, :attr:`cell_edges` and :attr:`cell_vertices`.
        """

        if dim2 >= dim1:
            raise ValueError("""dim2 must be less than dim1.""")
        if dim2 < 0:
            raise ValueError("""dim2 cannot be negative.""")
        if dim1 > self.dim:
            raise ValueError("""dim1 cannot exceed the mesh dimension.""")

        if dim1 == 1:
            if self.dim == 1:
                return self.cell_vertices
            else:
                return self.edge_vertices
        elif dim1 == 2:
            if dim2 == 0:
                return self.cell_vertices
            else:
                return self.cell_edges

    def jacobian(self, c):
        """Return the Jacobian matrix for the specified cell.

        :param c: The index of the cell(s) for which to return the Jacobian.
        :result: The Jacobian for cells ``c``.
        """

        # If we are working with a single cell:
        if isinstance(c, int):
            # Find the global vertex coordinates for this cell:
            xhat = self.vertex_coords[self.cell_vertices[c], :]
            # Since J_{a, b} = sum_j { (xhat_j)_a  * grad_b(basis_j)(X) } 
            # for any X, if we set X = 0, this is the definition of matrix
            # multiplication between xhat^T and self.grad_linear_basis,
            # so J is given by:
            return xhat.T @ self.grad_linear_basis
        else:
            # Find the vertices we are working with:
            vertex_indices = self.cell_vertices[c]
            # Find the global vertex coordinates for all of the cells:
            xhat = self.vertex_coords[vertex_indices.ravel(), :] \
                .reshape(vertex_indices.shape[0], self.dim + 1, self.dim)
            # As J_{c, a, b} = sum_j { (xhat_{j, c})_a * grad_b(basis_j)(X) } 
            # like above, we can represent it using einsum as (transposed)
            # matrix multiplication along the first axis:
            return np.einsum('ikj,kl->ijl', xhat, self.grad_linear_basis)


class UnitIntervalMesh(Mesh):
    """A mesh of the unit interval."""
    def __init__(self, nx):
        """
        :param nx: The number of cells.
        """
        points = np.array(list((x,) for x in np.linspace(0, 1, nx + 1)))
        points.shape = (points.shape[0], 1)

        cells = np.array(list((a, a+1) for a in range(nx)))

        super(UnitIntervalMesh, self).__init__(points,
                                               cells)


class UnitSquareMesh(Mesh):
    """A triangulated :class:`Mesh` of the unit square."""
    def __init__(self, nx, ny):
        """
        :param nx: The number of cells in the x direction.
        :param ny: The number of cells in the y direction.
        """
        points = list((x, y)
                      for x in np.linspace(0, 1, nx + 1)
                      for y in np.linspace(0, 1, ny + 1))

        mesh = Delaunay(points)

        super(UnitSquareMesh, self).__init__(mesh.points,
                                             mesh.simplices)

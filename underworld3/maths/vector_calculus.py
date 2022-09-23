import underworld3
import underworld3.timing as timing
import sympy


class mesh_vector_calculus:
    """Vector calculus on uw row matrices
    - this class is designed to augment the functionality of a mesh"""

    def __init__(self, mesh):
        self.mesh = mesh
        self.dim = self.mesh.dim

    def curl(self, matrix):
        """
        \( \nabla \cross \mathbf{v} \)

        Returns the curl of a 3D vector field or the out-of-plane
        component of a 2D vector field
        """

        vector = self.to_vector(matrix)
        vector_curl = sympy.vector.curl(vector)

        if self.dim == 3:
            return self.to_matrix(vector_curl)
        else:
            # if 2d, the out-of-plane vector is not defined in the basis so a scalar is returned (cf. vorticity)
            return vector_curl.dot(self.mesh.N.k)

    def divergence(self, matrix):
        """
        \( \nabla \cdot \mathbf{v} \)
        """
        vector = self.to_vector(matrix)
        scalar_div = sympy.vector.divergence(vector)
        return scalar_div

    def gradient(self, scalar):
        """
        $\nabla \phi$
        """

        if isinstance(scalar, sympy.Matrix) and scalar.shape == (1, 1):
            scalar = scalar[0, 0]

        vector_gradient = sympy.vector.gradient(scalar)
        return self.to_matrix(vector_gradient)

    def to_vector(self, matrix):

        if isinstance(matrix, sympy.vector.Vector):
            return matrix  # No need to convert

        if matrix.shape == (1, self.dim):
            vector = sympy.vector.matrix_to_vector(matrix, self.mesh.N)
        elif matrix.shape == (self.dim, 1):
            vector = sympy.vector.matrix_to_vector(matrix.T, self.mesh.N)
        elif matrix.shape == (1, 1):
            vector = matrix[0, 0]
        else:
            print(f"Unable to convert matrix of size {matrix.shape} to sympy.vector")
            vector = None

        return vector

    def to_matrix(self, vector):

        if isinstance(vector, sympy.Matrix) and vector.shape == (1, self.dim):
            return vector

        if isinstance(vector, sympy.Matrix) and vector.shape == (self.dim, 1):
            return vector.T

        matrix = sympy.Matrix.zeros(1, self.dim)
        base_vectors = self.mesh.N.base_vectors()

        for i in range(self.dim):
            matrix[0, i] = vector.dot(base_vectors[i])

        return matrix

    def jacobian(self, vector):

        jac = vector.diff(self.mesh.X).reshape(self.mesh.X.shape[1], vector.shape[1]).tomatrix().T

        return jac

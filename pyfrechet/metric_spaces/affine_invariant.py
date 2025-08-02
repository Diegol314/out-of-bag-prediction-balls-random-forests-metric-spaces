from geomstats.geometry.spd_matrices import SPDMatrices, SPDAffineMetric
from .riemannian_manifold import RiemannianManifold

class AffineInvariant(RiemannianManifold):
    """
    Affine-invariant Riemannian metric space for dxd SPD matrices.
    
    The distance between two SPD matrices A and B is defined as:
    d(A, B) = || log(A^(-1/2) B A^(-1/2)) ||_F

    The Fréchet mean of a set of SPD matrices {X_i} is estimated from the sample using the function mean_riemann from pyriemann.utils.mean, based on the papers:
    .. [1] `Principal geodesic analysis for the study of nonlinear statistics
        of shape
        <https://ieeexplore.ieee.org/document/1318725>`_
        P.T. Fletcher, C. Lu, S. M. Pizer, S. Joshi.
        IEEE Trans Med Imaging, 2004, 23(8), pp. 995-1005
    .. [2] `A differential geometric approach to the geometric mean of
        symmetric positive-definite matrices
        <https://epubs.siam.org/doi/10.1137/S0895479803436937>`_
        M. Moakher. SIAM J Matrix Anal Appl, 2005, 26 (3), pp. 735-747
    """
    def __init__(self, dim):
        manifold = SPDMatrices(n=dim)
        manifold.metric = SPDAffineMetric(space=manifold)
        super().__init__(manifold)

    def __str__(self):
        return f'SPD_matrices (affine invariant metric) (dim={self.manifold.n})'
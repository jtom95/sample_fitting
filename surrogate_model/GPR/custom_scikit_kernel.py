"""
==========================================================
Custom kernel for Gaussian Process Regression (GPR)
==========================================================
"""

# A new custom kernel can also be created once it subclasses
# :class:`sklearn.gaussian_process.kernels.Kernel` and once the abstract
# methods :meth:`__call__`, :meth:`diag`, and :meth:`is_stationary`
# are implemented.
#

# Here is a step-by-step construction of the Sigmoid Kernel
# .. math::
# k(x_i, x_j)= \tanh(\alpha x_i \cdot x_j + \sigma)
# First, :math:`\alpha` and :math:`\sigma` are read in as hyperparameters::

from typing import Union, List, Tuple
import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin
from scipy.spatial.distance import pdist, cdist, squareform


def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale


class StepKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, max_range=Union[Tuple[float, float], float], max_range_bounds="fixed"):
        # max_range_bounds = np.atleast_1d(max_range_bounds)
        self.max_range = max_range
        self.max_range_bounds = max_range_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.max_range) and len(self.max_range) > 1

    @property
    def hyperparameter_max_range(self):
        if self.anisotropic:
            return Hyperparameter(
                "max_range",
                "numeric",
                self.max_range_bounds,
                len(self.max_range),
                fixed=True,
            )
        return Hyperparameter("max_range", "numeric", self.max_range_bounds, fixed=True)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        max_range = _check_length_scale(X, self.max_range)
        if Y is None:
            if isinstance(self.max_range, (float, int)) or len(self.max_range) == 1:
                ndist = pdist(X, metric="euclidean")
                K = np.ones_like(ndist)
                K[ndist > max_range] = 0
            elif len(max_range) == 2:
                x_dist = pdist(X[:, 0:1], metric="euclidean")
                y_dist = pdist(X[:, 1:2], metric="euclidean")
                K = np.ones_like(x_dist)
                K[x_dist > max_range[0]] = 0
                K[y_dist > max_range[1]] = 0
            else:
                raise ValueError("max_range must be a scalar or a 2-tuple")

            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            if isinstance(self.max_range, (float, int)) or len(self.max_range) == 1:
                ndist = cdist(X, Y, metric="euclidean")
                K = np.ones_like(ndist)
                K[ndist > self.max_range] = 0
            elif len(max_range) == 2:
                x_dist = cdist(X[:, 0:1], Y[:, 0:1], metric="euclidean")
                y_dist = cdist(X[:, 1:2], Y[:, 1:2], metric="euclidean")
                K = np.ones_like(x_dist)
                K[x_dist > max_range[0]] = 0
                K[y_dist > max_range[1]] = 0

        if eval_gradient:
            if self.hyperparameter_max_range.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or max_range.shape[0] == 1:
                K_gradient = np.zeros_like(K)
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = np.zeros_like(K)
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(max_range=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.max_range)),
            )
        else:  # isotropic
            return "{0}(max_range={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.max_range)[0]
            )


class MyRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Radial basis function kernel (aka squared-exponential kernel).

    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length scale
    parameter :math:`l>0`, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:

    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)

    where :math:`l` is the length scale of the kernel and
    :math:`d(\\cdot,\\cdot)` is the Euclidean distance.
    For advice on how to set the length scale parameter, see e.g. [1]_.

    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    See [2]_, Chapter 4, Section 4.2, for further details of the RBF kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    .. [2] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8354..., 0.03228..., 0.1322...],
           [0.7906..., 0.0652..., 0.1441...]])
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), max_range=None):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.max_range = max_range

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            ndist = pdist(X, metric="euclidean")
            dists = pdist(X / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix

            # set elements with distances greater than max_range to zero
            if self.max_range is not None:
                K[ndist > (self.max_range)] = 0

            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            ndist = cdist(X, Y, metric="euclidean")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # set elements with distances greater than max_range to zero
            if self.max_range is not None:
                K[ndist > (self.max_range)] = 0

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (length_scale**2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )

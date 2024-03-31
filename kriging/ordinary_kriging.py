from typing import Tuple
import numpy as np
from scipy.spatial.distance import cdist


class OrdinaryKrigingEstimator:
    def __init__(self, variogram_model, data, sample_points):
        """
        Initialize the OrdinaryKrigingEstimator. This class is meant to work with 2D data.

        Args:
            variogram_model (dict): The fitted variogram model dictionary from VariogramAnalyzer.
            data (np.ndarray): The data values at the sample points.
            sample_points (np.ndarray): The coordinates of the sample points. The shape should be (N, 2).
        """
        if sample_points.shape[0] in [2, 3] and sample_points.shape[1] > 3:
            sample_points = sample_points.T
        if sample_points.shape[1] == 3:
            sample_points = sample_points[:, :2]
        if sample_points.ndim != 2 or sample_points.shape[1] != 2:
            raise ValueError("The sample_points should have shape (N, 2)")
        
        self.variogram_model = variogram_model
        self.sample_values = data
        self.sample_points = sample_points

    def _calculate_variogram(self, distances)-> np.ndarray:
        """
        Calculate the variogram values for given distances based on the fitted variogram model.

        Args:
            distances (np.ndarray): The distances for which to calculate variogram values.

        Returns:
            np.ndarray: The variogram values corresponding to the distances.
        """
        return self.variogram_model["variogram_generator"](distances)

    def _assemble_kriging_matrix(self, num_sample_points)-> np.ndarray:
        """
        Assemble the kriging matrix.

        Args:
            num_sample_points (int): The number of sample points.

        Returns:
            np.ndarray: The kriging matrix.
        """
        kriging_matrix = np.zeros((num_sample_points + 1, num_sample_points + 1))
        kriging_matrix[:num_sample_points, :num_sample_points] = -self._calculate_variogram(cdist(self.sample_points, self.sample_points))
        np.fill_diagonal(kriging_matrix, 0.0)
        kriging_matrix[num_sample_points, :] = 1.0
        kriging_matrix[:, num_sample_points] = 1.0
        kriging_matrix[num_sample_points, num_sample_points] = 0.0
        return kriging_matrix

    def _solve_kriging_vectorized(self, kriging_matrix, sample_estimation_distances, mask)-> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the kriging system as a vectorized operation.

        Args:
            kriging_matrix (np.ndarray): The kriging matrix.
            sample_estimation_distances (np.ndarray): The distances between the sample points and the estimation points.
            mask (np.ndarray): The mask array indicating the points to exclude.

        Returns:
            tuple: The estimated values and estimation variances.
        """
        num_estimation_points = sample_estimation_distances.shape[0]
        num_sample_points = self.sample_points.shape[0]
        estimation_matrix = np.zeros((num_estimation_points, num_sample_points + 1, 1))
        estimation_matrix[:, :num_sample_points, 0] = -self._calculate_variogram(sample_estimation_distances)
        estimation_matrix[:, num_sample_points, 0] = 1.0

        if (~mask).any():
            mask_estimation_matrix = np.repeat(mask[:, np.newaxis, np.newaxis], num_sample_points + 1, axis=1)
            estimation_matrix = np.ma.array(estimation_matrix, mask=mask_estimation_matrix)

        weights = np.linalg.solve(kriging_matrix, estimation_matrix.reshape((num_estimation_points, num_sample_points + 1)).T).reshape((1, num_sample_points + 1, num_estimation_points)).T
        estimated_values = np.sum(weights[:, :num_sample_points, 0] * self.sample_values, axis=1)
        estimation_variances = np.sum(weights[:, :, 0] * -estimation_matrix[:, :, 0], axis=1)

        return estimated_values, estimation_variances

    def _solve_kriging_loop(self, kriging_matrix, sample_estimation_distances, mask)-> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the kriging system by looping over all specified points.

        Args:
            kriging_matrix (np.ndarray): The kriging matrix.
            sample_estimation_distances (np.ndarray): The distances between the sample points and the estimation points.
            mask (np.ndarray): The mask array indicating the points to exclude.

        Returns:
            tuple: The estimated values and estimation variances.
        """
        num_estimation_points = sample_estimation_distances.shape[0]
        num_sample_points = self.sample_points.shape[0]
        estimated_values = np.zeros(num_estimation_points)
        estimation_variances = np.zeros(num_estimation_points)

        kriging_matrix_inv = np.linalg.inv(kriging_matrix)

        for i in np.nonzero(~mask)[0]:
            estimation_distances = sample_estimation_distances[i]
            estimation_vector = np.zeros((num_sample_points + 1, 1))
            estimation_vector[:num_sample_points, 0] = -self._calculate_variogram(estimation_distances)
            estimation_vector[num_sample_points, 0] = 1.0
            weights = np.dot(kriging_matrix_inv, estimation_vector)
            estimated_values[i] = np.sum(weights[:num_sample_points, 0] * self.sample_values)
            estimation_variances[i] = np.sum(weights[:, 0] * -estimation_vector[:, 0])

        return estimated_values, estimation_variances

    def estimate(self, estimation_points, n_closest_points=None, backend='vectorized')-> Tuple[np.ndarray, np.ndarray]:
        """
        Perform ordinary kriging estimation at the specified estimation points.

        Args:
            estimation_points (np.ndarray): The points at which to estimate the values.
            n_closest_points (int, optional): The number of nearby points to use in the calculation.
                This can speed up the calculation for large datasets, but should be used with caution.
                Default is None.
            backend (str, optional): Specifies the backend to use for kriging.
                Supported options are 'vectorized' (default) and 'loop'.

        Returns:
            tuple: The estimated values and estimation variances at the estimation points.
        """
        num_sample_points = self.sample_points.shape[0]
        num_estimation_points = estimation_points.shape[0]
        kriging_matrix = self._assemble_kriging_matrix(num_sample_points)

        mask = np.zeros(num_estimation_points, dtype="bool")

        if n_closest_points is not None:
            from scipy.spatial import cKDTree

            tree = cKDTree(self.sample_points)
            _, indices = tree.query(estimation_points, k=n_closest_points, eps=0.0)
            sample_estimation_distances = cdist(estimation_points, self.sample_points[indices])
        else:
            sample_estimation_distances = cdist(estimation_points, self.sample_points)

        if backend == 'vectorized':
            estimated_values, estimation_variances = self._solve_kriging_vectorized(kriging_matrix, sample_estimation_distances, mask)
        elif backend == 'loop':
            estimated_values, estimation_variances = self._solve_kriging_loop(kriging_matrix, sample_estimation_distances, mask)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return estimated_values, estimation_variances

    def estimate_grid(self, grid, n_closest_points=None, backend='vectorized')-> Tuple[np.ndarray, np.ndarray]:
        """
            Perform ordinary kriging estimation on a grid.

            Args:
                grid (np.ndarray): The grid points at which to estimate the values.
                    The grid should have shape (2, M, N), where the first dimension represents the x and y coordinates,
                    and M and N are the dimensions of the grid.
                n_closest_points (int, optional): The number of nearby points to use in the calculation.
                    This can speed up the calculation for large datasets, but should be used with caution.
                    Default is None.
                backend (str, optional): Specifies the backend to use for kriging.
                    Supported options are 'vectorized' (default) and 'loop'.
            Returns:
                tuple: The estimated values and estimation variances at the grid points.
                    The estimated values and variances will have shape (M, N).
        """
        if np.squeeze(grid).ndim == 2 and np.ndim(grid) == 4:
            grid = grid.squeeze()
        if grid.shape[0] == 3:
            grid = grid[:2]
        if grid.ndim != 3 or grid.shape[0] != 2:
            raise ValueError("The grid should have shape (2, M, N)")
        M, N = grid.shape[1], grid.shape[2]
        estimation_points = grid.reshape(2, -1).T
        estimated_values, estimation_variances = self.estimate(
            estimation_points, n_closest_points, backend
        )
        estimated_values = estimated_values.reshape(M, N)
        estimation_variances = estimation_variances.reshape(M, N)
        return estimated_values, estimation_variances

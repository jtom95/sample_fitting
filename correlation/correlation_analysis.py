from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor


# from my_packages.auxiliary_plotting_functions.composite_plots import stack_figures
from my_packages.classes.aux_classes import Grid

from ._correlation_plotter_mixin import CorrelationPlotterMixin

class CorrelationAnalyzer(CorrelationPlotterMixin):
    """
    A class for analyzing correlations in 3D data.

    Attributes:
        data (numpy.ndarray): The 3D data array with dimensions x, y, f.
        position_grid (Grid, optional): The position grid corresponding to the data.
        frequency_indices (List[int]): The list of frequency indices to analyze.
    """

    def __init__(
        self,
        data: np.ndarray,
        position_grid: Optional[Grid] = None,
        frequency_indices: Optional[List[int]] = None,
        normalize: bool = True,
        absolute_correlation_values: bool = True
    ):
        """
        Initialize the CorrelationAnalyzer.

        Args:
            data (numpy.ndarray): The 3D data array with dimensions x, y, f.
            position_grid (Grid, optional): The position grid corresponding to the data. Defaults to None.
            frequency_indices (List[int], optional): The list of frequency indices to analyze. Defaults to None.
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
        """
        if normalize:
            data = self._normalize_data(data)
        if np.ndim(data) == 2:
            self.data = data[:, :, np.newaxis]  
        elif np.ndim(data) == 3:
            self.data = data
        else:
            raise ValueError("Data must be 2D or 3D.")
        self.position_grid = position_grid
        self.frequency_indices = (
            frequency_indices if frequency_indices is not None else list(range(data.shape[2]))
        )

        self.frequency_indices_order_dict = {
            freq_index: ii for ii, freq_index in enumerate(self.frequency_indices)
        }
        self.absolute_correlation_values = absolute_correlation_values

        self.x_correlation_matrix = self.get_correlation_matrix(axis=1)
        self.y_correlation_matrix = self.get_correlation_matrix(axis=0)
        self.x_mean_matrix = self.get_mean_matrix(axis=1)
        self.y_mean_matrix = self.get_mean_matrix(axis=0)

    # alias
    @property
    def critical_indices(self) -> np.ndarray:
        return self.frequency_indices

    @property
    def combined_correlation_matrix(self) -> np.ndarray:
        combined_corr_list = []
        for ii, f_index in enumerate(self.frequency_indices):
            combined_corr = self.get_combined_correlation_matrix(index=f_index)
            combined_corr_list.append(combined_corr)
        return np.stack(combined_corr_list, axis=-1)

    def get_combined_correlation_matrix(self, index=None):
        if index is None:
            index = 0
        else:
            index = self.frequency_indices_order_dict.get(index, None)
            if index is None:
                raise ValueError("The frequency index is not in the list of frequency indices.")

        x_corr = self.x_correlation_matrix[..., index]
        y_corr = self.y_correlation_matrix[..., index]

        # Get the maximum size of the matrices
        max_size = max(x_corr.shape[0], y_corr.shape[0])

        # Pad the matrices with zeros to make them the same size
        x_corr_padded = np.pad(
            x_corr,
            ((0, max_size - x_corr.shape[0]), (0, max_size - x_corr.shape[1])),
            mode="constant",
            constant_values=np.nan
        )
        y_corr_padded = np.pad(
            y_corr,
            ((0, max_size - y_corr.shape[0]), (0, max_size - y_corr.shape[1])),
            mode="constant",
            constant_values=np.nan
        )

        # Create the combined correlation matrix
        combined_corr = np.triu(y_corr_padded) + np.tril(x_corr_padded)
        np.fill_diagonal(combined_corr, 0.5 * (np.diag(x_corr_padded) + np.diag(y_corr_padded)))

        return combined_corr

    @property
    def data_at_frequencies(self) -> np.ndarray:
        """Filter the data to include only the slices corresponding to the specified frequency indices."""
        return self.data[:, :, self.frequency_indices]

    @staticmethod
    def _normalize_data(data: np.ndarray) -> np.ndarray:
        """
        Normalize the data to the range [0, 1].

        Args:
            data (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The normalized data.
        """
        return (data - data.min()) / (data.max() - data.min())

    def get_correlation_matrix(self, axis: int = 0) -> np.ndarray:
        """
        Calculate the correlation matrix along the specified axis for each frequency index.

        Args:
            axis (int, optional): The axis along which to calculate the correlation. Defaults to 0.

        Returns:
            numpy.ndarray: The correlation matrices for each frequency index.

        Raises:
            ValueError: If the axis is not 0 or 1.
        """
        if axis not in [0, 1]:
            raise ValueError("Axis must be 0 or 1.")

        correlation_matrices = []
        for freq_index in self.frequency_indices:
            data = self.data[..., freq_index]
            df = pd.DataFrame(data)
            if axis == 0:
                correlation_matrix = df.corr()
            else:
                correlation_matrix = df.T.corr()
            correlation_matrices.append(correlation_matrix)
        correlation_matrix = np.stack(correlation_matrices, axis=-1)
        if self.absolute_correlation_values:
            # correlation_matrix = np.abs(correlation_matrix)
            correlation_matrix = np.clip(correlation_matrix, 0, 1)
        return correlation_matrix

    def get_mean_matrix(self, axis: int = 0) -> np.ndarray:
        """
        Calculate the mean matrix along the specified axis for each frequency index.

        Args:
            axis (int, optional): The axis along which to calculate the mean. Defaults to 0.

        Returns:
            numpy.ndarray: The mean matrices for each frequency index.
        """
        mean_matrices = []
        for freq_index in self.frequency_indices:
            data = self.data[..., freq_index]
            df = pd.DataFrame(data)
            mean_axis = df.max(axis=axis).to_numpy()
            N = len(mean_axis)
            mean_matrix = np.zeros((N, N))
            for ii in range(N):
                for jj in range(N):
                    mean_matrix[ii, jj] = np.mean([mean_axis[ii], mean_axis[jj]])
            mean_matrices.append(mean_matrix)

        mean_matrix = np.stack(mean_matrices, axis=-1)
        if self.absolute_correlation_values:
            # mean_matrix = np.abs(mean_matrix)
            mean_matrix = np.clip(mean_matrix, 0, 1)
        return mean_matrix

    def __repr__(self) -> str:
        return f"CorrelationAnalyzer(data={self.data.shape}, position_grid={self.position_grid})"

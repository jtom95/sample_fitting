from typing import Tuple, Optional, List
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from my_packages.auxiliary_plotting_functions.composite_plots import stack_figures
from my_packages.classes.aux_classes import Grid
from my_packages.registries.scan_registry import ScanRegistry


from process_scanner_measurements.registries2.registry import SamplePlaneRegistry
from process_scanner_measurements.sample_planes.sample_plane_operations import Transformations

from ._correlation_plotter_mixin import CorrelationPlotterMixin
from .correlation_analysis import CorrelationAnalyzer


class CorrelationAnalysis_SamplePlaneR(CorrelationPlotterMixin):
    def __init__(
        self,
        sample_planeR: SamplePlaneRegistry,
        frequency_indices: List[int],
        normalize=True,
    ):
        self.sample_planeR = sample_planeR
        self.normalize = normalize
        self.frequency_indices = frequency_indices

        self.frequency_indices_order_dict = {
            freq_index: ii for ii, freq_index in enumerate(self.frequency_indices)
        }
        self.position_grid = self.sample_planeR.sample_planes[0].ideal_grid

        self.x_correlation_matrix = None
        self.y_correlation_matrix = None
        self.x_mean_matrix = None
        self.y_mean_matrix = None

        self.logger = logging.getLogger(__name__)

        self.analyzers = self.create_analyzers()
    
    @property
    def frequency(self) -> np.ndarray:
        return self.sample_planeR.sample_planes[0].frequency
    
    @property
    def critical_frequencies(self) -> np.ndarray:
        return self.sample_planeR.sample_planes[0].frequency[self.frequency_indices]
    
    #alias
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

    def create_correlationR(self)-> ScanRegistry:
        correlationR = ScanRegistry()
        for ii, sample_plane in enumerate(self.sample_planeR.sample_planes):
            self.logger.info(f"{ii}/{len(self.sample_planeR.sample_planes)}) creating analyzer for sample_plane")
            signal_matrix = sample_plane.create_signal_matrix()
            analyzer = CorrelationAnalyzer(
                data=signal_matrix,
                position_grid=sample_plane.create_position_matrix(),
                frequency_indices=self.frequency_indices,
                normalize=self.normalize,
            )
            xcorrelation_matrix = analyzer.get_correlation_matrix(axis=0)
            ycorrelation_matrix = analyzer.get_correlation_matrix(axis=1)

            for f_index in self.frequency_indices:
                f_position = self.frequency_indices_order_dict[f_index]
                xcorr_scan_v = xcorrelation_matrix[:,:,f_position]
                ycorr_scan_v = ycorrelation_matrix[:,:,f_position]

                boilerplate_scan = Transformations.simple_Scan_at_index(
                    sample_plane=sample_plane,
                    frequency_index=f_index,
                )

                xaxis = self.position_grid.x
                yaxis = self.position_grid.y
                height = sample_plane.height

                xgrid = np.array(np.meshgrid(xaxis, xaxis, height, indexing="ij"))
                ygrid = np.array(np.meshgrid(yaxis, yaxis, height, indexing="ij"))

                xcorr_scan = boilerplate_scan.create_new(
                    scan=xcorr_scan_v,
                    grid=xgrid,
                ).add_tag("x_correlation")

                ycorr_scan = boilerplate_scan.create_new(
                    scan=ycorr_scan_v,
                    grid=ygrid,
                ).add_tag("y_correlation")

                correlationR.add(xcorr_scan)
                correlationR.add(ycorr_scan)
        return correlationR                

    def create_analyzers(self)-> List[CorrelationAnalyzer]:
        list_of_analyzers = []
        for ii, sample_plane in enumerate(self.sample_planeR.sample_planes):
            self.logger.info(f"{ii}/{len(self.sample_planeR.sample_planes)}) creating analyzer for sample_plane")
            signal_matrix = sample_plane.create_signal_matrix()
            analyzer = CorrelationAnalyzer(
                data=signal_matrix,
                position_grid=sample_plane.create_position_matrix(),
                frequency_indices=self.frequency_indices,
                normalize=self.normalize,
            )
            list_of_analyzers.append(analyzer)
        return list_of_analyzers

    def generate_scan_registry(self)-> ScanRegistry:
        list_of_scanR = []
        for ii, f_index in enumerate(self.frequency_indices):
            self.logger.info(f"{ii}/{len(self.frequency_indices)}) making scan at frequency index {f_index}")
            scanR_f0 = self.sample_planeR.to_ScanR(
                frequency_index=f_index,
                method="nearest",
            )
            list_of_scanR.append(scanR_f0)
        scanR = np.sum(list_of_scanR, axis=0)
        return scanR

    def run(self)-> "CorrelationAnalysis_SamplePlaneR":
        self.x_correlation_matrix = np.mean(
            [analyzer.x_correlation_matrix for analyzer in self.analyzers], axis=0
        )
        self.y_correlation_matrix = np.mean(
            [analyzer.y_correlation_matrix for analyzer in self.analyzers], axis=0
        )
        self.x_mean_matrix = np.mean(
            [analyzer.x_mean_matrix for analyzer in self.analyzers], axis=0
        )
        self.y_mean_matrix = np.mean(
            [analyzer.y_mean_matrix for analyzer in self.analyzers], axis=0
        )
        return self

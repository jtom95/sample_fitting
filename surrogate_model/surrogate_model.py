from typing import Tuple, Optional


import numpy as np
import matplotlib.pyplot as plt

from .abstract_sample_model import AbstractSampleModel

from my_packages.classes.aux_classes import Grid
from my_packages.EM_fields.scans import Scan

class SurrogateModel:
    def __init__(self, model: AbstractSampleModel):
        if not isinstance(model, AbstractSampleModel):
            raise TypeError("model must be an instance of Model")
        self.model = model

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)
    
    def prediction_std(self, X) -> np.ndarray:
        return self.model.prediction_std(X)
    
    def std_prediction_scan(self, grid: Grid, frequency: float = None, **kwargs) -> Scan:
        if frequency is None:
            frequency = -1

        shape2d = (grid.shape[1], grid.shape[2])
        points = grid.create_position_matrix()
        points2d = points[:, 0:2]
        predictions = self.prediction_std(points2d)
        predictions = predictions.reshape(shape2d)
        return Scan(predictions, grid=grid, freq=frequency, **kwargs)
    
    
    def predict_scan(self, grid: Grid, frequency: float = None, **kwargs) -> Tuple[Scan, Scan]:
        if frequency is None:
            frequency = -1

        shape2d = (grid.shape[1], grid.shape[2])
        points = grid.create_position_matrix()
        points2d = points[:, 0:2]
        predictions = self.predict(points2d)
        predictions = predictions.reshape(shape2d)
        return Scan(predictions, grid=grid, freq=frequency, **kwargs)  

    def score(self, X, y):
        # calculate the score as the
        # 1 - (sum of squared residuals / sum of squared total)
        
        # calculate the sum of squared residuals
        residuals = y - self.predict(X)
        ss_res = np.sum(residuals**2)
        
        # calculate the sum of squared total
        ss_tot = np.sum((y - np.mean(y))**2)
        
        # calculate the score
        score = 1 - (ss_res / ss_tot)
        
        return score
    
    @staticmethod
    def compare_scan(
        scan: Scan,
        fitting_points: np.ndarray,
        fitting_labels: np.ndarray,
        std_scan: Optional[Scan] = None,
        ax: Optional[np.ndarray] = None,
        units=None,
        **kwargs
    ):
        # overall max between fitting labels and labels_scan
        max_value = np.max([np.max(fitting_labels), np.max(scan.v)])
        min_value = np.min([np.min(fitting_labels), np.min(scan.v)])

        if ax is None:
            n_axes = 3 if std_scan is not None else 2
            def_height = 3 if std_scan is not None else 5
            fig, ax = plt.subplots(1, n_axes, figsize=(10, def_height), constrained_layout=True)
        else:
            fig = ax[0].get_figure()

        ax[0].scatter(
            fitting_points[:, 0],
            fitting_points[:, 1],
            c=fitting_labels,
            vmin=min_value,
            vmax=max_value,
            cmap="jet",
        )
        ax[0].set_xlabel("x (mm)")
        ax[0].set_ylabel("y (mm)")

        # transfrom to mm
        ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(x * 1e3)))
        ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(x * 1e3)))

        ax[0].set_title("Original Points")

        scan.plot(ax=ax[1], vmin=min_value, vmax=max_value, units=units, **kwargs)
        ax[1].set_title("Predicted Points")
        if std_scan is not None:
            std_scan.plot(ax=ax[2], units="std", **kwargs)
            ax[2].set_title("Standard Deviation")
        return fig, ax

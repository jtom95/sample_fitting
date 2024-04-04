from typing import Tuple, Optional, List, Dict, Callable, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

from scipy.optimize import curve_fit, least_squares
from scipy.spatial.distance import cdist


from .variogram_models import VariogramModels
from ._variogram_analysis_plotter_mixin import VariogramAnalyzerPlottingMixin

# from my_packages.auxiliary_plotting_functions.composite_plots import stack_figures
from my_packages.classes.aux_classes import Grid

from typing import Tuple, Optional, List, Dict, Callable, Literal

import numpy as np
from scipy.optimize import curve_fit

from .variogram_models import VariogramModels
from ._variogram_analysis_plotter_mixin import VariogramAnalyzerPlottingMixin


class VariogramFitting:
    def fit_variogram(
        self,
        model_type: Optional[Literal["gaussian", "exponential", "spherical", "hole_effect"]] = None,
        nugget: Optional[float] = None,
        sill: Optional[float] = None,
        range_: Optional[float] = None
    ) -> Dict:
        """
        Fit variogram models to the empirical variogram data and return the best-fitting model.

        Args:
            model_type (Optional[Literal["gaussian", "exponential", "spherical", "hole_effect"]]): The type of variogram model to fit.
            nugget (Optional[float]): The fixed nugget value. If provided, the nugget will not be optimized.
            sill (Optional[float]): The fixed sill value. If provided, the sill will not be optimized.
            range_ (Optional[float]): The fixed range value. If provided, the range will not be optimized.

        Returns:
            Dict: The fitted variogram model parameters for the best-fitting model.
        """
        if model_type is not None:
            return self._fit_variogram_model_type(model_type, nugget, sill, range_)
        model_types = ["gaussian", "exponential", "spherical", "hole_effect"]
        best_model = None
        min_residuals = float("inf")

        for model_type in model_types:
            model = self._fit_variogram_model_type(model_type, nugget, sill, range_)
            residuals = np.sum(model["residuals"] ** 2)

            if residuals < min_residuals:
                if model.get("nugget") is not None and model.get("nugget") < 0:
                    continue
                min_residuals = residuals
                best_model = model

        return best_model
    

    def _fit_variogram_model_type(
        self,
        model_type: Literal["gaussian", "exponential", "spherical", "hole_effect"],
        nugget: Optional[float] = None,
        sill: Optional[float] = None,
        range_: Optional[float] = None
    ) -> Dict:
        """Fit a variogram model to the empirical variogram data."""
        if self.lags is None or self.semivariances is None:
            raise ValueError(
                "Empirical variogram data not available. Call calculate_empirical_variogram() first."
            )

        model_functions = {
            "gaussian": VariogramModels.gaussian,
            "exponential": VariogramModels.exponential,
            "spherical": VariogramModels.spherical,
            "hole_effect": VariogramModels.hole_effect,
        }

        if model_type not in model_functions:
            raise ValueError(
                f"Invalid model_type. Available options: {', '.join(model_functions.keys())}"
            )

        model_function = model_functions[model_type]

        fitted_params = self._calculate_variogram_model(
            self.lags,
            self.semivariances,
            # model_type,
            model_function,
            weight=True,
            nugget=nugget,
            sill=sill,
            range_=range_
        )

        return_dict = {
            "nugget": fitted_params[0],
            "sill": fitted_params[1],
            "range_": fitted_params[2],
            "model_type": model_type,
        }

        def variogram_model(lag):
            return model_function(lag, *fitted_params)

        return_dict["model_function"] = model_function
        return_dict["variogram_generator"] = variogram_model
        return_dict["residuals"] = self._variogram_residuals(
            fitted_params,
            self.lags,
            self.semivariances,
            model_function,
            weight=True,
        )

        return return_dict

    @classmethod
    def _calculate_variogram_model(
        cls,
        lags,
        semivariance,
        variogram_function,
        weight,
        nugget=None,
        sill=None,
        range_=None
    ):
        """Function that fits a variogram model when parameters are not specified."""
        if sill is not None:
            sill = float(sill)
        if range_ is not None:
            range_ = float(range_)
        if nugget is not None:
            nugget = float(nugget)
        
        
        x0 = [
            np.amin(semivariance) if nugget is None else nugget,
            np.amax(semivariance) - np.amin(semivariance) if sill is None else sill,
            0.8 * np.amax(lags) if range_ is None else range_,
        ]
        bnds = (
            [0.0, 0.0, 0.0],
            [np.amax(semivariance), 10.0 * np.amax(semivariance), 1.5 * np.amax(lags)],
        )

        # Create a mask for the fixed parameters
        fixed_mask = np.array([nugget is not None, sill is not None, range_ is not None])

        # Create arrays for the initial values and bounds
        x0_arr = np.array(x0)
        bnds_arr = np.array(bnds)

        # Create a partial function for _variogram_residuals with fixed parameters
        residuals_func = partial(
            cls._variogram_residuals,
            lags=lags,
            semivariance=semivariance,
            variogram_function=variogram_function,
            weight=weight,
            nugget=nugget,
            sill=sill,
            range_=range_
        )

        # Create a new objective function that takes only the non-fixed parameters
        def objective(params):
            final_params = x0_arr.copy()
            final_params[~fixed_mask] = params
            return residuals_func(final_params)

        # Remove fixed parameters from initial values and bounds
        x0_opt = x0_arr[~fixed_mask]
        bnds_opt = np.array(tuple(zip(bnds_arr[0][~fixed_mask], bnds_arr[1][~fixed_mask]))).T

        res = least_squares(
            objective,
            x0_opt,
            bounds=bnds_opt,
            loss="soft_l1",
        )

        # Insert optimized parameters back into the final parameters
        final_params = x0_arr.copy()
        final_params[~fixed_mask] = res.x

        return final_params

    @classmethod
    def _variogram_residuals(cls, params, lags, semivariance, variogram_function, weight, nugget=None, sill=None, range_=None):
        """Function used in variogram model estimation."""
        nugget_, sill_, range__ = params
        if nugget is not None:
            nugget_ = nugget
        if sill is not None:
            sill_ = sill
        if range_ is not None:
            range__ = range_

        if weight:
            drange = np.amax(lags) - np.amin(lags)
            k = 2.1972 / (0.1 * drange)
            x0 = 0.7 * drange + np.amin(lags)
            weights = 1.0 / (1.0 + np.exp(-k * (x0 - lags)))
            weights /= np.sum(weights)
            resid = (variogram_function(lags, nugget_, sill_, range__) - semivariance) * weights
        else:
            resid = variogram_function(lags, nugget_, sill_, range__) - semivariance
        return resid

    # def _fit_variogram_model_type(
    #     self,
    #     model_type: Literal["linear", "power", "gaussian", "exponential", "spherical", "hole_effect"],

    # ) -> Dict:
    #     """Fit a variogram model to the empirical variogram data."""
    #     if self.lags is None or self.semivariances is None:
    #         raise ValueError(
    #             "Empirical variogram data not available. Call calculate_empirical_variogram() first."
    #         )

    #     model_functions = {
    #         "linear": VariogramModels.linear,
    #         "power": VariogramModels.power,
    #         "gaussian": VariogramModels.gaussian,
    #         "exponential": VariogramModels.exponential,
    #         "spherical": VariogramModels.spherical,
    #         "hole_effect": VariogramModels.hole_effect,
    #     }

    #     if model_type not in model_functions:
    #         raise ValueError(
    #             f"Invalid model_type. Available options: {', '.join(model_functions.keys())}"
    #         )

    #     model_function = model_functions[model_type]

    #     fitted_params = self._calculate_variogram_model(
    #         self.lags,
    #         self.semivariances,
    #         model_type,
    #         model_function,
    #         weight=True,
    #     )

    #     if model_type == "linear":
    #         return_dict = {
    #             "nugget": fitted_params[0],
    #             "slope": fitted_params[1],
    #             "model_type": model_type,
    #         }
    #     elif model_type == "power":
    #         return_dict = {
    #             "nugget": fitted_params[0],
    #             "scale": fitted_params[1],
    #             "exponent": fitted_params[2],
    #             "model_type": model_type,
    #         }
    #     else:
    #         return_dict = {
    #             "nugget": fitted_params[0],
    #             "sill": fitted_params[1],
    #             "range_": fitted_params[2],
    #             "model_type": model_type,
    #         }

    #     def variogram_model(lag):
    #         return model_function(lag, *fitted_params)

    #     return_dict["model_function"] = model_function
    #     return_dict["variogram_generator"] = variogram_model
    #     return_dict["residuals"] = self._variogram_residuals(
    #         fitted_params,
    #         self.lags,
    #         self.semivariances,
    #         model_function,
    #         weight=True,
    #     )

    #     return return_dict

    # @classmethod
    # def _calculate_variogram_model(
    #     cls,
    #     lags,
    #     semivariance,
    #     variogram_model,
    #     variogram_function,
    #     weight,
    # ):
    #     """Function that fits a variogram model when parameters are not specified."""
    #     if variogram_model == "linear":
    #         x0 = [
    #             np.amin(semivariance),
    #             (np.amax(semivariance) - np.amin(semivariance)) / (np.amax(lags) - np.amin(lags)),
    #         ]
    #         bnds = ([0.0, 0.0], [np.amax(semivariance), np.inf])
    #     elif variogram_model == "power":
    #         x0 = [
    #             np.amin(semivariance),
    #             (np.amax(semivariance) - np.amin(semivariance)) / (np.amax(lags) - np.amin(lags)),
    #             1.1,
    #         ]
    #         bnds = ([0.0, 0.001, 0.0], [np.amax(semivariance), np.inf, 1.999,])
    #     else:
    #         x0 = [
    #             np.amin(semivariance),
    #             np.amax(semivariance) - np.amin(semivariance),
    #             0.8 * np.amax(lags),
    #         ]
    #         bnds = (
    #             [0.0, 0.0, 0.0],
    #             [np.amax(semivariance), 10.0 * np.amax(semivariance), 1.5 * np.amax(lags)],
    #         )

    #     res = least_squares(
    #         cls._variogram_residuals,
    #         x0,
    #         bounds=bnds,
    #         loss="soft_l1",
    #         args=(lags, semivariance, variogram_function, weight),
    #     )
    #     return res.x

    # @classmethod
    # def _variogram_residuals(cls, params, x, y, variogram_function, weight):
    #     """Function used in variogram model estimation."""
    #     if weight:
    #         drange = np.amax(x) - np.amin(x)
    #         k = 2.1972 / (0.1 * drange)
    #         x0 = 0.7 * drange + np.amin(x)
    #         weights = 1.0 / (1.0 + np.exp(-k * (x0 - x)))
    #         weights /= np.sum(weights)
    #         resid = (variogram_function(x, *params) - y) * weights
    #     else:
    #         resid = variogram_function(x, *params) - y
    #     return resid


class VariogramAnalyzer(VariogramAnalyzerPlottingMixin, VariogramFitting):
    """
    A class for analyzing variograms in 2D data.

    Attributes:
        positions (numpy.ndarray): The 2D array of positions with shape (N, 2).
        values (numpy.ndarray): The 1D array of measured values with shape (N,).
    """

    def __init__(
        self,
        positions: np.ndarray,
        values: np.ndarray,
    ):
        """
        Initialize the VariogramAnalyzer.

        Args:
            positions (numpy.ndarray): The 2D array of positions with shape (N, 2).
            values (numpy.ndarray): The 1D array of measured values with shape (N,).
        """
        if positions.shape[0] != values.shape[0]:
            raise ValueError("The number of positions and values must be the same.")

        self.positions = positions
        self.values = values

        self.lags = None
        self.semivariances = None
        self.angles_ = None
        self.tolerances_ = None
        self.deltas_ = None
        
    def reset(self):
        self.lags = None
        self.semivariances = None
        self.angles_ = None
        self.tolerances_ = None
        self.deltas_ = None

    def calculate_empirical_variogram(
        self,
        lag_step: float,
        angle: Optional[float]=None,
        tolerance: Optional[float]=None,
        max_range: Optional[float]=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the variogram data.

        Args:
            lag_step (float): The lag step size.
            angle (float, optional): The angle for the variogram.
            tolerance (float): The tolerance for the variogram.
            max_range (float, optional): The maximum range for the variogram.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The lags and semivariances.
        """
        
        self.angles_ = []
        self.tolerances_ = []
        self.deltas_ = []
        
        lags, semivariances = self._calculate_variogram_data(lag_step, angle, tolerance, max_range)

        self.lags = np.array(lags)
        self.semivariances = np.array(semivariances)
        
        return self.lags, self.semivariances

    def _calculate_variogram_data(
        self,
        lag_step: float,
        angle: Optional[float]=None,
        tolerance: Optional[float]=None,
        max_range: Optional[float] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate the variogram data.

        Args:
            lag_step (float): The lag step size.
            angle (float, optional): The angle for the variogram.
            tolerance (float): The tolerance for the variogram.
            max_range (float, optional): The maximum range for the variogram.

        Returns:
            Tuple[List[float], List[float]]: The lags and semivariances.
        """
        lags = []
        semivariances = []
        current_lag = lag_step

        for semivariance in self.semivariances_generator(
            lag_step=lag_step,
            angle=angle,
            tolerance=tolerance,
            max_range=max_range,
        ):
            lags.append(current_lag)
            semivariances.append(semivariance)
            current_lag += lag_step

        return lags, semivariances

    def semivariances_generator(
        self,
        lag_step=1,
        angle: float | int | str = None,
        tolerance=5,
        max_range: int | float = None,
        increase_lag_after=None,
        increase_lag_by=3,
        # weighted=False,
        deg=True,
    ):
        """
        Generate semivariances.

        Args:
            lag_step (float, optional): The lag step size. Defaults to 1.
            angle (float | int | str, optional): The angle for the variogram. Defaults to None.
            tolerance (float, optional): The tolerance for the variogram. Defaults to 5.
            max_range (int | float, optional): The maximum range for the variogram. Defaults to None.
            increase_lag_after (int, optional): The number of lags after which to increase the lag step. Defaults to None.
            increase_lag_by (int, optional): The factor by which to increase the lag step. Defaults to 3.
            deg (bool, optional): Whether the angle is in degrees. Defaults to True.

        Yields:
            float: The semivariance for each lag.
        """
        if isinstance(angle, str):
            if angle.capitalize() == "X":
                angle = 0
            elif angle.capitalize() == "Y":
                angle = 90 if deg else np.pi / 2
            else:
                raise ValueError("angle must be None, 'x', 'y' or a float")

        if deg and angle is not None:
            angle = np.deg2rad(angle)
            tolerance = np.deg2rad(tolerance)

        if max_range is None:
            max_range_at_angle = np.max(np.sqrt(np.sum(self.positions**2, axis=1)))
        else:
            max_range_at_angle = max_range

        if increase_lag_after is False or increase_lag_after is None:
            num_lags_at_angle = max_range_at_angle / lag_step
            num_lags_at_angle = int(np.ceil(num_lags_at_angle))
        else:
            num_with_min_lag = max_range_at_angle / lag_step
            after_min_lag = num_with_min_lag - increase_lag_after
            num_with_increased_lag = after_min_lag / increase_lag_by
            num_lags_at_angle = increase_lag_after + num_with_increased_lag
            num_lags_at_angle = int(np.ceil(num_lags_at_angle))

        for i in range(num_lags_at_angle):
            if increase_lag_after is not False and increase_lag_after is not None:
                if i < increase_lag_after:
                    lag_at_step = lag_step * (i + 1)
                    delta = (lag_step * 0.5, lag_step * 0.5)
                else:
                    lag_at_step = (
                        increase_lag_by * lag_step * (i + 1 - increase_lag_after)
                        + lag_step * increase_lag_after
                    )
                    delta = (increase_lag_by * lag_step * 0.5, increase_lag_by * lag_step * 0.5)
            else:
                lag_at_step = lag_step * (i + 1)
                delta = (lag_step * 0.5, lag_step * 0.5)
            semivariance = self.calculate_semivariance_at_lag(
                lag=lag_at_step,
                delta=delta,
                angle=angle,
                tolerance=tolerance,
                # max_range=max_range,
                deg=False,
                # weighted=weighted,
            )
            yield semivariance

    def calculate_semivariance_at_lag(
        self,
        lag,
        delta: Tuple[float, float] = None,
        angle=None,
        tolerance=5,
        consider_both_signs=True,
        # max_range=False,
        deg=True,
        # weighted=False,
    ):
        """
        Calculate the semivariance for a specific lag.

        Args:
            lag (float): The lag distance.
            delta (Tuple[float, float], optional): The lag tolerance. Defaults to None.
            angle (float, optional): The angle for the variogram. Defaults to None.
            tolerance (float, optional): The tolerance for the variogram. Defaults to 5.
            consider_both_signs (bool, optional): Whether to consider both positive and negative angles. Defaults to True.
            max_range (float, optional): The maximum range for the variogram. Defaults to False.
            deg (bool, optional): Whether the angle is in degrees. Defaults to True.
            weighted (bool, optional): Whether to use weighted semivariances. Defaults to False.

        Returns:
            float: The semivariance for the specified lag.
        """
        valid_pairs = self.get_pairs_at_lag(
            lag, delta, angle, tolerance, consider_both_signs, deg, register_parameters=True)

        # if max_range:
        #     distances = cdist(self.positions, self.positions)
        #     valid_pairs &= distances <= max_range

        D1, D2 = np.meshgrid(self.values, self.values)
        
        valid_squared_differences = (D1[valid_pairs] - D2[valid_pairs]) ** 2

        # if weighted:
        #     weights = (np.abs(D1[valid_pairs]) + np.abs(D2[valid_pairs])) / 2
        #     weighted_semivariances = valid_squared_differences * weights
        #     return (
        #         np.average(weighted_semivariances, weights=weights)
        #         if weighted_semivariances.size > 0
        #         else 0
        #     )
        semivariance = np.mean(valid_squared_differences)/2 if valid_squared_differences.size > 0 else 0
        return semivariance
    
    def get_pairs_at_lag(
        self,
        lag: float,
        delta: Tuple[float, float] = None,
        angle: Optional[float] = None,
        tolerance: float = 5,
        consider_both_signs: bool = True,
        deg: bool = True,
        register_parameters: bool = False,
    ) -> np.ndarray:
        """
        Get pairs of points that fall within a specified lag distance and optionally within a specified angle and tolerance.

        Args:
            lag (float): The lag distance.
            delta (Tuple[float, float], optional): The lag tolerance. Defaults to (lag * 0.5, lag * 0.5).
            angle (float, optional): The angle for the variogram. Defaults to None.
            tolerance (float, optional): The tolerance for the variogram. Defaults to 5.
            consider_both_signs (bool, optional): Whether to consider both positive and negative angles. Defaults to True.
            deg (bool, optional): Whether the angle is in degrees. Defaults to True.

        Returns:
            np.ndarray: A boolean array indicating valid pairs.
        """
        if delta is None:
            delta = (lag * 0.5, lag * 0.5)
        elif isinstance(delta, (int, float)):
            delta = (delta * 0.5, delta * 0.5)

        distances = cdist(self.positions, self.positions)
        valid_pairs = (distances > (lag - delta[0])) & (distances <= (lag + delta[1]))

        if angle is not None:
            if deg:
                angle = np.deg2rad(angle)
                tolerance = np.deg2rad(tolerance)

            angles = np.arctan2(
                self.positions[:, 1, np.newaxis] - self.positions[:, 1],
                self.positions[:, 0, np.newaxis] - self.positions[:, 0],
            )
            valid_angle_pairs = np.abs(angles - angle) <= tolerance
            if consider_both_signs:
                opposite_angle = angle - np.pi
                if opposite_angle == -np.pi:
                    opposite_angle = np.pi
                valid_angle_pairs |= np.abs(angles - opposite_angle) <= tolerance

            valid_pairs &= valid_angle_pairs
        
        if register_parameters:
            # keep track of the deltas and angles
            if self.deltas_ is not None:
                self.deltas_.append(delta)
            if self.angles_ is not None:
                self.angles_.append(np.rad2deg(angle) if angle is not None else None)
            if self.tolerances_ is not None:
                self.tolerances_.append(np.rad2deg(tolerance) if angle is not None else None)

        return valid_pairs

    def __repr__(self) -> str:
        """
        Return a string representation of the VariogramAnalyzer.

        Returns:
            str: The string representation of the VariogramAnalyzer.
        """
        return f"VariogramAnalyzer(positions={self.positions.shape}, values={self.values.shape})"

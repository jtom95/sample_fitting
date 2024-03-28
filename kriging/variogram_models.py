""""
Variogram models for fitting. Taken from PyKrige [1]_.
"""
from dataclasses import dataclass
import numpy as np
@dataclass
class VariogramModels:
    @staticmethod
    def linear(d, slope, nugget):
        """Linear model, m is [slope, nugget]"""
        return slope * d + nugget

    @staticmethod
    def power(d, scale, exponent, nugget):
        """Power model, m is [scale, exponent, nugget]"""
        return scale * d**exponent + nugget

    @staticmethod
    def gaussian(d, nugget, sill, range_):
        """Gaussian model, m is [psill, range, nugget]"""
        return sill * (1.0 - np.exp(-(d**2.0) / (range_ * 4.0 / 7.0) ** 2.0)) + nugget

    @staticmethod
    def exponential(d, nugget, sill, range_):
        """Exponential model, m is [psill, range, nugget]"""
        return sill * (1.0 - np.exp(-d / (range_ / 3.0))) + nugget

    @staticmethod
    def spherical(d, nugget, sill, range_):
        """Spherical model, m is [psill, range, nugget]"""
        return np.piecewise(
            d,
            [d <= range_, d > range_],
            [
                lambda x: sill * ((3.0 * x) / (2.0 * range_) - (x**3.0) / (2.0 * range_**3.0))
                + nugget,
                sill + nugget,
            ],
        )

    @staticmethod
    def hole_effect(d, nugget, sill, range_):
        """Hole Effect model, m is [psill, range, nugget]"""

        return sill * (1.0 - (1.0 - d / (range_ / 3.0)) * np.exp(-d / (range_ / 3.0))) + nugget

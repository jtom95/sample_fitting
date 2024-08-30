from typing import Tuple, Optional, List, Union
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, Matern
import matplotlib.pyplot as plt
import numpy as np

from ..GPR_light.gpr_surrogate_model import GPR
from .find_kernel_from_dipole_pattern import DipoleFieldKernelExtractor

from my_packages.registries.scan_registry import ScanRegistry
from my_packages.EM_fields.scans import Scan, ComplexScan, FieldScan
from my_packages.constants import MeasuremeableFields
from my_packages.auxiliary_plotting_functions.composite_plots import stack_figures

class GPRonScanR:
    # Class constants for default values
    DEFAULT_C_LEVEL = 1.0
    DEFAULT_C_LEVEL_BOUNDS = (1e-3, 1e3)
    DEFAULT_WHITE_LEVEL = 1e-5
    DEFAULT_WHITE_LEVEL_BOUNDS = (1e-10, 5)

    def __init__(
        self, 
        scanR:ScanRegistry, 
        height: float, 
        frequency: Optional[float] = None, 
        alpha=1e-10, 
        n_restarts_optimizer=5
        ):

        self.scanR = scanR
        self.height = height
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.Ez = None
        self.Hx = None
        self.Hy = None

        self.gpr_Ez = None
        self.gpr_Hx = None  
        self.gpr_Hy = None

        self._check_scanR()
        self._extract_scans(frequency=frequency)

    @property
    def noise_level_Ez(self):
        noise_key = None
        # find key with "noise_level" in it
        for key in self.gpr_Ez.gp.kernel_.get_params():
            if "noise_level" in key and "bounds" not in key:
                noise_key = key
                break
        if noise_key is None:
            return None
        return self.gpr_Ez.gp.kernel_.get_params()[noise_key]

    @property
    def noise_level_Hx(self):
        noise_key = None
        # find key with "noise_level" in it
        for key in self.gpr_Hx.gp.kernel_.get_params():
            if "noise_level" in key and "bounds" not in key:
                noise_key = key
                break
        if noise_key is None:
            return None
        return self.gpr_Hx.gp.kernel_.get_params()[noise_key]

    @property
    def noise_level_Hy(self):
        noise_key = None
        # find key with "noise_level" in it
        for key in self.gpr_Hy.gp.kernel_.get_params():
            if "noise_level" in key and "bounds" not in key:
                noise_key = key
                break
        if noise_key is None:
            return None
        return self.gpr_Hy.gp.kernel_.get_params()[noise_key]

    def get_noise_levels(self, dB=False):
        """
        Returns the noise power of the surrogate model. That is the variance of the additive white noise in the measurements.
        In order to get a value with the same units as the fields you can use the sqrt
        """
        if dB:
            return 10*np.log10(self.noise_level_Ez), 10*np.log10(self.noise_level_Hx), 10*np.log10(self.noise_level_Hy)
        return self.noise_level_Ez, self.noise_level_Hx, self.noise_level_Hy

    def get_noise_std(self):    
        """
        Returns the standard deviation of the noise in the measurements. This can be useful because it is the same units as the fields
        """
        return np.sqrt(self.noise_level_Ez), np.sqrt(self.noise_level_Hx), np.sqrt(self.noise_level_Hy)

    def _check_scanR(self):
        if self.scanR is None:
            raise ValueError("The scan registry is not set")
        if isinstance(self.scanR, list):
            if all(isinstance(scan, (Scan, ComplexScan, FieldScan)) for scan in self.scanR):
                self.scanR = ScanRegistry(self.scanR)
            else:
                raise ValueError("The scans in the registry are not of the correct type")

        # check all three scans are in the registry
        if not all(
            quid in self.scanR.quids
            for quid in [MeasuremeableFields.Ez, MeasuremeableFields.Hx, MeasuremeableFields.Hy]
        ):
            raise ValueError("The registry does not contain all required scans")

    def _extract_scans(self, frequency: Optional[float] = None):
        if frequency is not None:
            self.scanR = self.scanR.getR(frequency=frequency)
        scanR_h = self.scanR.getR(height=self.height)
        self.Ez = scanR_h.get(quid="Ez")[0].to_Scan()
        self.Hx = scanR_h.get(quid="Hx")[0].to_Scan()
        self.Hy = scanR_h.get(quid="Hy")[0].to_Scan()
        if frequency is None:
            if self.Ez.f != self.Hx.f or self.Ez.f != self.Hy.f:
                raise ValueError("The scans do not have the same frequency")
        return self.Ez, self.Hx, self.Hy

    def fit_gprs_with_fixed_kernel(self, kernel: Optional[Union[RBF, C]] = None) -> Tuple[GPR, GPR, GPR]:
        gpr_Ez = GPR(
            kernel=kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts_optimizer
        )
        gpr_Hx = GPR(
            kernel=kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts_optimizer
        )
        gpr_Hy = GPR(
            kernel=kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts_optimizer
        )

        gpr_Ez.fit_Scan(self.Ez.to_Scan())
        gpr_Hx.fit_Scan(self.Hx.to_Scan())
        gpr_Hy.fit_Scan(self.Hy.to_Scan())

        self.gpr_Ez = gpr_Ez
        self.gpr_Hx = gpr_Hx
        self.gpr_Hy = gpr_Hy

        return gpr_Ez, gpr_Hx, gpr_Hy

    def fit_gprs_with_adaptive_kernel(
        self,
        kernel_extractor: DipoleFieldKernelExtractor,
        include_structural_correlation_estimation: bool = False,
        estimate_noise: bool = False,
        estimate_C: bool = False,
        C_level: Optional[float] = None,
        C_level_bounds: Optional[Tuple[float, float]] = None,
        white_level: Optional[float] = None,
        white_level_bounds: Optional[Tuple[float, float]] = None,
    ) -> Tuple[GPR, GPR, GPR]:
        # Use default values if parameters are not provided
        C_level = C_level if C_level is not None else self.DEFAULT_C_LEVEL
        C_level_bounds = (
            C_level_bounds if C_level_bounds is not None else self.DEFAULT_C_LEVEL_BOUNDS
        )
        white_level = white_level if white_level is not None else self.DEFAULT_WHITE_LEVEL
        white_level_bounds = (
            white_level_bounds
            if white_level_bounds is not None
            else self.DEFAULT_WHITE_LEVEL_BOUNDS
        )

        if (
            kernel_extractor.Ez_kernel is None
            or kernel_extractor.Hx_kernel is None
            or kernel_extractor.Hy_kernel is None
        ):
            raise ValueError(
                "The kernel extractor does not have all required kernels. You must first extract the kernels"
            )

        simple_Ez_kernel = kernel_extractor.Ez_kernel
        simple_Hx_kernel = kernel_extractor.Hx_kernel
        simple_Hy_kernel = kernel_extractor.Hy_kernel

        if include_structural_correlation_estimation:
            simple_Ez_kernel = simple_Ez_kernel + C(
                constant_value=0.3, constant_value_bounds=(1e-5, 1)
            ) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
            simple_Hx_kernel = simple_Hx_kernel + C(
                constant_value=0.3, constant_value_bounds=(1e-5, 1)
            ) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
            simple_Hy_kernel = simple_Hy_kernel + C(
                constant_value=0.3, constant_value_bounds=(1e-5, 1)
            ) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        
        Ez_kernel = simple_Ez_kernel
        Hx_kernel = simple_Hx_kernel
        Hy_kernel = simple_Hy_kernel
        
        if estimate_C:
            Ez_kernel = C(
                C_level, (C_level_bounds[0], C_level_bounds[1])
            ) * simple_Ez_kernel
            Hx_kernel = C(
                C_level, (C_level_bounds[0], C_level_bounds[1])
            ) * simple_Hx_kernel
            Hy_kernel = C(
                C_level, (C_level_bounds[0], C_level_bounds[1])
            ) * simple_Hy_kernel
        
        if estimate_noise:
            Ez_kernel = Ez_kernel + WhiteKernel(
                white_level, (white_level_bounds[0], white_level_bounds[1])
            )
            
            Hx_kernel = Hx_kernel + WhiteKernel(
                white_level, (white_level_bounds[0], white_level_bounds[1])
            )
            
            Hy_kernel = Hy_kernel + WhiteKernel(
                white_level, (white_level_bounds[0], white_level_bounds[1])
            )

        gpr_Ez = GPR(
            kernel = Ez_kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts_optimizer
        )
        gpr_Hx = GPR(
            kernel = Hx_kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts_optimizer
        )
        gpr_Hy = GPR(
            kernel = Hy_kernel, alpha = self.alpha, n_restarts_optimizer=self.n_restarts_optimizer
        )

        gpr_Ez.fit_Scan(self.Ez.to_Scan())
        gpr_Hx.fit_Scan(self.Hx.to_Scan())
        gpr_Hy.fit_Scan(self.Hy.to_Scan())

        self.gpr_Ez = gpr_Ez
        self.gpr_Hx = gpr_Hx
        self.gpr_Hy = gpr_Hy

        return gpr_Ez, gpr_Hx, gpr_Hy

    def compare_fitting(self, new_scanR: ScanRegistry):
        fig_Ez, ax = self.gpr_Ez.compare_with_scan(
            scan=new_scanR.getR(height=self.height).get(quid="Ez")[0].to_Scan(), include_samples=True
        )
        fig_Ez.suptitle(f"$E_z$ component", fontsize=16)

        fig_Hx, ax = self.gpr_Hx.compare_with_scan(
            scan=new_scanR.getR(height=self.height).get(quid="Hx")[0].to_Scan(), include_samples=True
        )
        fig_Hx.suptitle(f"$H_x$ component", fontsize=16)
        fig_Hy, ax = self.gpr_Hy.compare_with_scan(
            scan=new_scanR.getR(height=self.height).get(quid="Hy")[0].to_Scan(), include_samples=True
        )
        fig_Hy.suptitle(f"$H_y$ component", fontsize=16)

        fig = stack_figures([fig_Ez, fig_Hx, fig_Hy], figshape="column", figsize=(8, 6), aspect="equal")
        fig[0].suptitle(f"Surrogate models at height {self.height*1e3:.0f} mm", fontsize=12, y=1.03)

        return fig

    def compare_fitting1(self, new_scanR: ScanRegistry, figsize=(8, 6), include_samples=True, quid_labels_fontsize=12, artistic=False, gpr_shape=(50, 50)):
        width_ratios = [0.05] + [1]*3
        fig, ax = plt.subplots(3, 4, figsize=figsize, constrained_layout=True, width_ratios=width_ratios)
        self.gpr_Ez.compare_with_scan(
            scan=new_scanR.getR(height=self.height).get(quid="Ez")[0].to_Scan(), include_samples=include_samples, ax=ax[0, 1:], shape=gpr_shape
        )
        self.gpr_Hx.compare_with_scan(
            scan = new_scanR.getR(height=self.height).get(quid="Hx")[0].to_Scan(), include_samples= include_samples, ax=ax[1, 1:], shape=gpr_shape
        )
        self.gpr_Hy.compare_with_scan(
            scan = new_scanR.getR(height=self.height).get(quid="Hy")[0].to_Scan(), include_samples= include_samples, ax=ax[2, 1:], shape=gpr_shape
        )

        for axx, quid in zip(ax[:, 0], ["Ez", "Hx", "Hy"]):
            axx.axis("off")
            axx.text(
                0.5,
                0.5,
                quid, 
                fontsize=quid_labels_fontsize,
                ha="center",
                va="center",
                rotation="horizontal",
            )

        if artistic:
            for axx in ax[:, 1:].flatten():
                # remove labels and ticks
                axx.axis("off")

        return fig, ax

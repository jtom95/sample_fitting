from dataclasses import dataclass
import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from typing import Literal


from spectrum_analysis.spatial_fft.multi_scan import FFT2DMultiScanLowLevel 

from ._noise_power_extractor_plotting import NoisePowerExtractorPlotMixin

@dataclass
class NoisePowerExtractorConfigs:
    number_of_dc_points: int = 1
    cutoff_distance_ratio: bool = 0.1
    envelope_type: Literal["step", "linear"] = "linear"
    envelope_transition_range: int = 2
    fft_normalization_type: Literal["ortho", "forward", "backward"] = "ortho"
    margin_ratio: float = 0.1


class NoisePowerExtractor(NoisePowerExtractorPlotMixin):
    def __init__(
        self,
        data: np.ndarray,
        number_of_dc_points=1,
        cutoff_distance_ratio: bool = 0.1,
        number_of_highest_points=0,
        envelope_type: Literal["step", "linear"] = "step",
        envelope_transition_range=2,
        fft_normalization_type: Literal["ortho", "forward", "backward"] = "ortho",
        margin_ratio: float = 0.1,
    ):
        """
        Initialize the NoisePowerExtractor class with input data.

        Args:
            data (np.ndarray): The input data array with shape (X, Y, F), where F is the number of scans.
            number_of_dc_points (int): The number of points to consider as DC component (default: 1).
            energy_threshold_fraction (float): The fraction of total energy to use as the threshold (default: 0.7).
            number_of_highest_points (int): The number of highest points to remove (default: 0).
            envelope_transition_range (int): The range of the linear transition for the envelope (default: 5).
            fft_normalization_type (Literal["ortho", "forward", "backward"]): The normalization type for FFT (default: "ortho").
        """

        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        if data.ndim != 3:
            raise ValueError("Input data must be a 3D array with shape (X, Y, F).")

        self.data = data
        self.num_scans = data.shape[2]

        self.envelope_type = envelope_type
        self.number_of_dc_points = number_of_dc_points
        self.number_of_highest_points = number_of_highest_points
        self.envelope_transition_range = envelope_transition_range
        self.fft_normalization_type = fft_normalization_type
        self.margin_ratio = margin_ratio
        self.cutoff_distance_ratio = cutoff_distance_ratio

        self.m_ = None
        self.M_ = None

    # @property
    # def scale_(self):
    #     if not hasattr(self, "m_") or not hasattr(self, "M_") or self.m_ is None or self.M_ is None:
    #         return None
    #     return self.M_ - self.m_

    def _normalize_data(self):
        """
        Normalize the input data along axes 0 and 1 so that all X, Y scans are between 0 and 1.
        """
        m = np.min(self.data, axis=(0, 1))
        M = np.max(self.data, axis=(0, 1))
        self.m_ = m
        self.M_ = M

        self.mean_ = np.mean(self.data, axis=(0, 1))
        self.scale_ = np.std(self.data, axis=(0, 1))

        self.normalized_data = (self.data - self.mean_) / self.scale_

    def _init_fft_analyzer(self, use_normalized_data: bool = True):
        self.fft_analyzer = FFT2DMultiScanLowLevel(
            self.normalized_data if use_normalized_data else self.data,
            number_of_dc_points=self.number_of_dc_points,
            number_of_highest_points=self.number_of_highest_points,
            envelope_transition_range=self.envelope_transition_range,
            fft_normalization_type=self.fft_normalization_type,
        )

    def get_cutoff_distance(self):
        """
        Get the cutoff distance for each scan using the FFT2DMultiScanLowLevel class.

        Returns:
            np.ndarray: The cutoff distance for each scan with shape (F,).
        """
        if not hasattr(self, "fft_analyzer"):
            self._init_fft_analyzer(use_normalized_data=True)
        if isinstance(self.cutoff_distance_ratio, (int, float)):
            # transform the cutoff distance ratio to the cutoff distance
            cutoff_distance = self.cutoff_distance_ratio * np.min(self.data.shape[:2])
            cutoff_distance = cutoff_distance * np.ones(self.num_scans)
        elif isinstance(self.cutoff_distance_ratio, (list, np.ndarray)):
            if len(self.cutoff_distance_ratio) != self.num_scans:
                raise ValueError("set_cutoff_distance must have the same length as the number of scans.")
            cutoff_distance = np.array(self.cutoff_distance_ratio)*np.min(self.data.shape[:2])

        else: 
            raise ValueError("set_cutoff_distance must be a boolean or a number.")

        return cutoff_distance

    def make_envelope(self, cutoff_distance: np.ndarray = None):
        """
        Make the envelope for each scan using the FFT2DMultiScanLowLevel class.

        Args:
            cutoff_distance (np.ndarray): The cutoff distance for each scan with shape (F,).

        Returns:
            np.ndarray: The envelope for each scan with shape (X, Y, F).
        """
        if not hasattr(self, "fft_analyzer"):
            self._init_fft_analyzer(use_normalized_data=True)

        if cutoff_distance is None:
            cutoff_distance = self.get_cutoff_distance()

        # Generate the step envelope
        if self.envelope_type == "linear":
            envelope = self.fft_analyzer.generate_linear_envelope(cutoff_distance)
        else:
            envelope = self.fft_analyzer.generate_step_envelope(cutoff_distance)
        return envelope

    def separate_frequencies(self, envelope: np.ndarray = None, normalize: bool = False):
        """
        Separate the frequencies for each scan using the FFT2DMultiScanLowLevel class.

        Args:
            envelope (np.ndarray): The envelope for each scan with shape (X, Y, F).
            normalize (bool): Whether to normalize the frequencies (default: False).

        Returns:
            Tuple[np.ndarray, np.ndarray]: The low and high frequencies for each scan with shape (X, Y, F).
        """
        if not hasattr(self, "fft_analyzer"):
            self._init_fft_analyzer(use_normalized_data=True)

        if envelope is None:
            # Find the cutoff distance
            cutoff_distance = self.get_cutoff_distance()
            envelope = self.make_envelope(cutoff_distance)

        # Separate the frequencies
        low_freq, high_freq = self.fft_analyzer.separate_frequencies(envelope, normalize=normalize)
        return low_freq, high_freq

    def estimate_noise_powers(self, normalize: bool = True, method: Literal["fft", "ifft"] = "ifft"):
        """
        Estimate the noise powers for each scan using the FFT2DMultiScanLowLevel class.

        Returns:
            np.ndarray: The estimated noise powers for each scan with shape (F,).
        """
        if normalize:
            # Normalize the data
            self._normalize_data()
            self._init_fft_analyzer(use_normalized_data=True)

        cutoff_distance = self.get_cutoff_distance()
        envelope = self.make_envelope(cutoff_distance)

        if method == "fft":
            # Estimate the noise powers
            noise_powers = np.zeros(self.num_scans)
            for i in range(self.num_scans):
                data_of_interest = self.fft_analyzer.fft_data[envelope[:, :, i] == 0, i].flatten()

                # remove the average
                noise_power_estimate = np.std(data_of_interest)**2
                noise_powers[i] = noise_power_estimate

            return noise_powers * self.scale_ ** 2
        elif method == "ifft":
            # Separate the frequencies
            low_freq, high_freq = self.separate_frequencies(envelope, normalize=False)

            # remove the mean from high_frequency _ as it does not affect the noise power
            high_freq = high_freq - np.mean(high_freq, axis=(0, 1))

            reconstructed = self.fft_analyzer.ifft_transform(high_freq)

            # remove the margins as these can be affected by the edge effects
            marginx = np.round(self.margin_ratio * reconstructed.shape[0]).astype(int)
            marginy = np.round(self.margin_ratio * reconstructed.shape[1]).astype(int)
            reconstructed = reconstructed[
                marginx:-marginx,
                marginy:-marginy,
                :,
            ]

            return np.mean(reconstructed**2, axis=(0, 1)) * self.scale_**2

    def estimate_signal_powers(self, normalize: bool = True, method: Literal["fft", "ifft"] = "ifft"):
        """
        Estimate the signal powers for each scan using the FFT2DMultiScanLowLevel class.

        Returns:
            np.ndarray: The estimated signal powers for each scan with shape (F,).
        """
        if normalize:
            # Normalize the data
            self._normalize_data()
            self._init_fft_analyzer(use_normalized_data=True)

        cutoff_distance = self.get_cutoff_distance()
        envelope = self.make_envelope(cutoff_distance)

        if method == "fft":
            # Estimate the noise powers
            signal_powers = np.zeros(self.num_scans)
            for i in range(self.num_scans):
                data_of_interest = self.fft_analyzer.fft_data[envelope[:, :, i] != 0, i].flatten()

                # remove the average
                signal_power_estimate = np.std(data_of_interest)**2
                signal_powers[i] = signal_power_estimate

            return signal_powers * self.scale_ ** 2
        elif method == "ifft":
            # Separate the frequencies
            low_freq, high_freq = self.separate_frequencies(envelope, normalize=False)

            # remove the mean from high_frequency _ as it does not affect the noise power
            low_freq = low_freq - np.mean(low_freq, axis=(0, 1))

            reconstructed = self.fft_analyzer.ifft_transform(low_freq)

            # remove the margins as these can be affected by the edge effects
            marginx = np.round(self.margin_ratio * reconstructed.shape[0]).astype(int)
            marginy = np.round(self.margin_ratio * reconstructed.shape[1]).astype(int)
            reconstructed = reconstructed[
                marginx:-marginx,
                marginy:-marginy,
                :,
            ]

            return np.mean(reconstructed**2, axis=(0, 1)) * self.scale_**2

    def estimate_snr(self, normalize: bool = True, method: Literal["fft", "ifft"] = "ifft", dB: bool = True):
        """
        Estimate the signal-to-noise ratio for each scan using the FFT2DMultiScanLowLevel class.

        Returns:
            np.ndarray: The estimated signal-to-noise ratio for each scan with shape (F,).
        """
        noise_powers = self.estimate_noise_powers(normalize=normalize, method=method)
        signal_powers = self.estimate_signal_powers(normalize=normalize, method=method)
        
        # add a small value to avoid division by zero
        noise_powers = np.clip(noise_powers, 1e-10, None)

        snr = signal_powers / noise_powers
        
        # clip the values to 1:
        snr = np.clip(snr, 1, None)

        if dB:
            return 10 * np.log10(snr)
        return snr

    def generate_configs(self):
        return NoisePowerExtractorConfigs(
            number_of_dc_points=self.number_of_dc_points,
            cutoff_distance_ratio=self.cutoff_distance_ratio,
            envelope_type=self.envelope_type,
            envelope_transition_range=self.envelope_transition_range,
            fft_normalization_type=self.fft_normalization_type,
            margin_ratio=self.margin_ratio,
        )
        
    @classmethod
    def from_configs(cls, data: np.ndarray, configs: NoisePowerExtractorConfigs):
        return cls(
            data,
            number_of_dc_points=configs.number_of_dc_points,
            cutoff_distance_ratio=configs.cutoff_distance_ratio,
            envelope_type=configs.envelope_type,
            envelope_transition_range=configs.envelope_transition_range,
            fft_normalization_type=configs.fft_normalization_type,
            margin_ratio=configs.margin_ratio,
        )

from typing import Tuple, Optional, List

import dill
from pathlib import Path

from .surrogate_model import SurrogateModel
from .abstract_sample_model import AbstractSampleModel


from my_packages.constants import MeasuremeableFields
from my_packages.EM_fields.scans import Scan, Grid





class RichSurrogateModel(SurrogateModel):
    extension = "rsurr_model"
    std_margin: int = 1

    def __init__(
        self,
        model: AbstractSampleModel,
        fitting_time: float = None,
        frequency: float = None,
        height: float = None,
        quid: MeasuremeableFields = MeasuremeableFields.UNK,
    ):
        super().__init__(model)
        if isinstance(quid, str):
            quid = MeasuremeableFields(quid)
        self.fitting_time = fitting_time
        self.frequency = frequency
        self.height = height
        self.quid = quid

    def predict_scan_and_std(self, grid: Grid, tag=None) -> Tuple[Scan, Scan]:
        """
        This method predicts scan and std with a single call to the surrogate model.
        """
        grid = grid.resample_at_height(self.height)
        frequency = self.frequency
        field_type = self.quid.value[0]
        component = self.quid.value[1]
        tags = tag
        pred, std = super().predict_scan_and_std(
            grid=grid, frequency=frequency, field_type=field_type, component=component, tags=tags
        )

        # the margin of the standard deviation is not reliable
        std_cropped = std.crop_n_pixels(
            x=(self.std_margin, self.std_margin), y=(self.std_margin, self.std_margin)
        )
        return pred, std_cropped

    def predict_scan(self, grid: Grid, tag=None) -> Scan:
        """
        This method predicts scan and std with a single call to the surrogate model.
        """
        grid = grid.resample_at_height(self.height)
        frequency = self.frequency
        field_type = self.quid.value[0]
        component = self.quid.value[1]
        tags = tag
        pred = super().predict_scan(
            grid=grid, frequency=frequency, field_type=field_type, component=component, tags=tags
        )
        return pred

    def save(self, filepath: Path) -> None:
        """
        Save the surrogate model to a file.
        Parameters
        ----------
        filepath : Path
            The file path to save the surrogate model.
        """
        # if the suffix is present, remove it
        if filepath.suffix != f".{self.extension}":
            filepath = filepath.with_suffix(f".{self.extension}")

        # check if the directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as file:
            dill.dump(self, file)

    @classmethod
    def load(cls, filepath: Path) -> "RichSurrogateModel":
        """
        Load the surrogate model from a file.
        Parameters
        ----------
        filepath : Path
            The file path to load the surrogate model.
        Returns
        -------
        RichSurrogateModel
            The loaded surrogate model.
        """
        # check if the suffix is present
        if filepath.suffix != f".{cls.extension}":
            filepath = filepath.with_suffix(f".{cls.extension}")
        with open(filepath, "rb") as file:
            surrogate_model = dill.load(file)
        return surrogate_model

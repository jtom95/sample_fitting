import os
from typing import Tuple
from enum import Enum


class TFUnits:
    def __init__(self, Nominator: Tuple[Enum], Denominator: Tuple[Enum]):
        if isinstance(Nominator, Enum):
            Nominator = (Nominator,)
        if isinstance(Denominator, Enum):
            Denominator = (Denominator,)
        self.Nominator = Nominator
        self.Denominator = Denominator

    def return_string(self):
        repr = ""
        for unit in self.Nominator:
            repr += unit.name
        repr += "/"
        for unit in self.Denominator:
            repr += unit.name
        return repr

    def factor(self):
        nominator_factor = 1
        for unit in self.Nominator:
            nominator_factor *= unit.value

        denominator_factor = 1
        for unit in self.Denominator:
            denominator_factor *= unit.value

        return nominator_factor / denominator_factor

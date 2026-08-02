from pathlib import Path

import numpy as np
import pandas as pd


class equations:
    def __init__(self):
        self.limits = {}

    def csv_path(self, file_name: str):
        this_file = Path.cwd().resolve()
        while this_file != this_file.parent:  #! stops if c:\ or /
            file = this_file / file_name
            if file.exists():
                return file
            else:
                this_file = this_file.parent

        raise FileNotFoundError(
            f"could not fined {file_name} in current project directory !"
        )

    def ALLOWED_RANGE(self, data, test=None):

        x = data.copy()
        series = isinstance(x, pd.Series)

        if series:
            x = x.to_frame()

        for col in x.columns:
            if not test:
                Q1 = x[col].quantile(0.25)
                Q3 = x[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR
                x[col] = np.clip(x[col], lower_limit, upper_limit)
                self.limits[col] = (lower_limit, upper_limit)

            else:
                low, high = self.limits[col]
                x[col] = np.clip(x[col], low, high)

        if series:
            x = x.squeeze()
        return x

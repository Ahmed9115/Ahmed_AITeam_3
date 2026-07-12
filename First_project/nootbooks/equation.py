import numpy as np
import pandas as pd


class equations:
    def __init__(self):
        self.limits = {}
        self.min_values = {}

    #! this fuction is from ai which is the implementation of the above mathematical formula to generate the distance in kilometers !
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in kilometers (use 3959 for miles)
        r = 6371
        return c * r

    #! this function is from ai which is calculation of the copass heading angle
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

        # Calculate bearing in degrees
        bearing = np.arctan2(x, y)
        bearing_degrees = np.degrees(bearing)

        # Normalize to 0-360 degrees compass heading
        return (bearing_degrees + 360) % 360

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

    def extract_min(self, x, negative_columns):
        data = x.copy()
        for col in negative_columns:
            self.min_values[col] = data[col].min()
        return self.min_values

    def negative_shift(self, x, columns, min_values):
        data = x.copy()
        for col in columns:
            data[col] = data[col] - min_values[col] + 1
        return data

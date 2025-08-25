from darts import TimeSeries
from darts.dataprocessing.transformers import InvertibleMapper, Scaler, Diff
from darts.dataprocessing.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from typing import Sequence



class Normalizer():
    def __init__(self):
        self.log_mapper = InvertibleMapper(
            fn=np.log,
            inverse_fn=np.exp, # type: ignore
            name="Log"
        )
        self.pipeline = Pipeline([self.log_mapper])

    def log_normalize(self, series: TimeSeries):
        return self.pipeline.fit_transform(series)
    
    def inverse_log_normalize(self, series: TimeSeries):
        return self.pipeline.inverse_transform(series)

class MyScaler():
    def __init__(self):
        self.scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))

    def fit_transform(self, series: TimeSeries | Sequence[TimeSeries]):
        return self.scaler.fit_transform(series)

    def transform(self, series: TimeSeries | Sequence[TimeSeries]):
        return self.scaler.transform(series)

    def inverse(self, series: TimeSeries | list[TimeSeries] | list[list[TimeSeries]]):
        return self.scaler.inverse_transform(series)

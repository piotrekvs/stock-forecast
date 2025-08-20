from darts import TimeSeries
from darts.dataprocessing.transformers import InvertibleMapper, Scaler, Diff
from darts.dataprocessing.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from typing import Sequence



class Normalizer():
    def __init__(self, diff=False):
        log_mapper = InvertibleMapper(
            fn=np.log,
            inverse_fn=np.exp, # type: ignore
            name="Log"
        )

        self.pipeline = Pipeline([log_mapper, Diff(lags=1)] if diff else [log_mapper])

    def log_normalize(self, series: TimeSeries) -> TimeSeries | Sequence[TimeSeries]:
        return self.pipeline.fit_transform(series)
    
    def inverse_log_normalize(self, series: TimeSeries) -> TimeSeries | Sequence[TimeSeries]:
        return self.pipeline.inverse_transform(series)

class Scalers():
    def __init__(self):
        # self.scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        self.scaler = Scaler(StandardScaler())

    def fit_scaler(self, series: TimeSeries | Sequence[TimeSeries]) -> TimeSeries | Sequence[TimeSeries]:
        return self.scaler.fit_transform(series)

    def transform(self, series: TimeSeries | Sequence[TimeSeries]) -> TimeSeries | Sequence[TimeSeries]:
        return self.scaler.transform(series)

    def inverse_scaler(self, series: TimeSeries | list[TimeSeries] | list[list[TimeSeries]]) -> (TimeSeries | list[TimeSeries] | list[list[TimeSeries]]):
        return self.scaler.inverse_transform(series)

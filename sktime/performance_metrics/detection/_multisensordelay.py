"""Multi-sensor detection delay for time series event detection."""
import numpy as np

from sktime.performance_metrics.detection._base import BaseDetectionMetric
from sktime.performance_metrics.detection._detectiondelaymean import (
    DetectionDelayMean,
)


class MultiSensorDetectionDelay(BaseDetectionMetric):
    r"""Detection delay metric that works across multiple sensor channels.

    A thin wrapper around ``DetectionDelayMean`` for cases where you have
    more than one sensor stream and want a single number that summarises
    how quickly the *system* (not just one sensor) caught an event.
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": True,
    }

    def __init__(
        self,
        sensor_cols=None,
        aggfunc="mean",
        sensor_weights=None,
        early_tolerance=0,
        max_delay=None,
    ):
        self.sensor_cols = sensor_cols
        self.aggfunc = aggfunc
        self.sensor_weights = sensor_weights
        self.early_tolerance = early_tolerance
        self.max_delay = max_delay
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute aggregated multi-sensor detection delay.

        Parameters
        ----------
        y_true : pd.DataFrame
            True event locations; one column per sensor.
        y_pred : pd.DataFrame
            Predicted event locations; same columns as ``y_true``.
        X : ignored

        Returns
        -------
        float
            Aggregated delay across all sensor channels.
        """
        if self.sensor_cols is None:
            raise ValueError(
                "sensor_cols cannot be None - pass a list of column names, "
                "e.g. sensor_cols=['vibration_ilocs', 'acoustic_ilocs']."
            )

        base = DetectionDelayMean(
            early_tolerance=self.early_tolerance,
            max_delay=self.max_delay,
        )

        scores = []
        for col in self.sensor_cols:
            y_t = y_true[[col]].rename(columns={col: "ilocs"}).dropna()
            y_p = y_pred[[col]].rename(columns={col: "ilocs"}).dropna()
            scores.append(base(y_t, y_p))

        if self.aggfunc == "min":
            return float(np.min(scores))
        if self.aggfunc == "max":
            return float(np.max(scores))
        if self.aggfunc == "weighted":
            if self.sensor_weights is None:
                raise ValueError(
                    "sensor_weights must be set when aggfunc='weighted'."
                )
            return float(np.average(scores, weights=self.sensor_weights))
        # default: "mean"
        return float(np.mean(scores))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Ignored here; included for API compatibility.

        Returns
        -------
        list of dict
        """
        return [
            {
                "sensor_cols": ["vibration_ilocs", "acoustic_ilocs"],
                "aggfunc": "mean",
            },
            {
                "sensor_cols": [
                    "vibration_ilocs",
                    "acoustic_ilocs",
                    "mechanical_ilocs",
                ],
                "aggfunc": "min",
                "early_tolerance": 10,
                "max_delay": 100,
            },
        ]
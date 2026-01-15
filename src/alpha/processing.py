from typing import Dict

import numpy as np


class SignalProcessor:
    @staticmethod
    def z_score(forecasts: Dict[int, float]) -> Dict[int, float]:
        if not forecasts:
            return {}
        vals = np.array(list(forecasts.values()))
        mean = np.mean(vals)
        std = np.std(vals)
        if std == 0:
            return {k: 0.0 for k in forecasts.keys()}
        return {k: (v - mean) / std for k, v in forecasts.items()}

    @staticmethod
    def winsorize(
        forecasts: Dict[int, float], limit: float = 3.0
    ) -> Dict[int, float]:
        return {k: max(min(v, limit), -limit) for k, v in forecasts.items()}

    @staticmethod
    def apply_decay(
        forecast: float,
        initial_time: float,
        current_time: float,
        half_life: float,
    ) -> float:
        """
        Applies exponential decay to a signal.
        Signal_t = Signal_0 * e^(-lambda * delta_t)
        where lambda = ln(2) / half_life
        """
        delta_t = current_time - initial_time
        if delta_t < 0:
            return forecast
        decay_constant = np.log(2) / half_life
        return float(forecast * np.exp(-decay_constant * delta_t))
